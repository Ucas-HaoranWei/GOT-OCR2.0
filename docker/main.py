"""
References:
    https://huggingface.co/ucaslcl/GOT-OCR2_0
    https://huggingface.co/spaces/ucaslcl/GOT_online/blob/main/app.py
    https://github.com/Ucas-HaoranWei/GOT-OCR2.0
"""

import asyncio
import base64
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Tuple, Optional, Any

import gradio as gr
from diskcache import Cache
from huggingface_hub import snapshot_download
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import AutoModel, AutoTokenizer

ROOT_DIR = Path(__file__).parent
CACHE_DIR = ROOT_DIR.joinpath(".cache")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    RUNTIME_ENV: str = "dev"
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 7860

    MODEL_NAME: str = "ucaslcl/GOT-OCR2_0"
    HF_HOME: str = "~/.cache/huggingface/"

    # fixme unavailable field
    #  "modeling_GOT.py", line 215, in forward --> self.embed_tokens(input_ids)
    # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:2 and cuda:0!
    # (when checking argument for argument index in method wrapper_CUDA__index_select)
    DEVICE: str = "cuda:0"

    UPLOAD_FOLDER: str = str(CACHE_DIR.joinpath("uploads"))
    RESULTS_FOLDER: str = str(CACHE_DIR.joinpath("results"))
    DISKCACHE_FOLDER: str = str(CACHE_DIR.joinpath("memory"))
    FILE_EXPIRATION_TIME: int = 3600

    def model_post_init(self, __context: Any) -> None:
        for folder in [self.UPLOAD_FOLDER, self.RESULTS_FOLDER, self.DISKCACHE_FOLDER]:
            Path(folder).mkdir(parents=True, exist_ok=True)

        current_time = time.time()
        for folder_ in [self.UPLOAD_FOLDER, self.RESULTS_FOLDER]:
            for path in Path(folder_).glob("**/*"):
                try:
                    if current_time - path.stat().st_mtime > self.FILE_EXPIRATION_TIME:
                        if path.is_file():
                            path.unlink()
                        elif path.is_dir():
                            shutil.rmtree(path)
                except Exception as err:
                    logger.error(f"Error while cleaning up {path}: {str(err)}")


def init_log(**sink_channel):
    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()

    persistent_format = (
        "<g>{time:YYYY-MM-DD HH:mm:ss.ms}</g> | "
        "<lvl>{level}</lvl>    | "
        "<c><u>{name}</u></c>:{function}:{line} | "
        "{message} - "
        "{extra}"
    )
    stdout_format = (
        "<g>{time:YYYY-MM-DD HH:mm:s.ms}</g> | "
        "<lvl>{level:<8}</lvl>    | "
        "<c>{name}</c>:<c>{function}</c>:<c>{line}</c> | "
        "<n>{message}</n>"
    )

    logger.remove()
    logger.add(
        sink=sys.stdout, colorize=True, level=log_level, format=stdout_format, diagnose=False
    )
    if sink_channel.get("error"):
        logger.add(
            sink=sink_channel.get("error"),
            level="ERROR",
            rotation="1 day",
            encoding="utf8",
            diagnose=False,
        )
    if sink_channel.get("runtime"):
        logger.add(
            sink=sink_channel.get("runtime"),
            level="TRACE",
            rotation="1 day",
            encoding="utf8",
            diagnose=False,
        )
    if sink_channel.get("serialize"):
        logger.add(
            sink=sink_channel.get("serialize"),
            level="DEBUG",
            format=persistent_format,
            encoding="utf8",
            diagnose=False,
            serialize=True,
        )
    logger.success(f"reset log level - LOG_LEVEL={log_level}")
    return logger


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper():
                result__ = await func(*args, **kwargs)
                end_time__ = time.time()
                execution_time__ = end_time__ - start_time
                logger.info(f"{func.__name__} execute: {execution_time__:.4f}s")
                return result__

            return async_wrapper()
        else:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"{func.__name__} execute: {execution_time:.4f}s")
            gc.collect()
            return result

    return wrapper


def load_model(model_name, device: str = "cuda"):
    try:
        logger.debug(f"Loading snapshot - {model_name}")
        model_cache = snapshot_download(model_name)

        logger.debug("Initializing tokenizer")
        _tokenizer = AutoTokenizer.from_pretrained(
            model_cache, trust_remote_code=True, device_map=device
        )

        logger.debug("Initializing model")
        _model = AutoModel.from_pretrained(
            model_cache,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=device,
            use_safetensors=True,
        )
        _model = _model.eval().to(device)

        logger.info(f"Model cache path: {model_cache}")
        logger.info(f"Model loaded on device: {device}")

        return _tokenizer, _model

    except Exception as err:
        logger.error(f"Error loading model: {str(err)}")
        raise


def get_file_hash(file_path: str) -> str:
    """Calculate the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# == Init == #
init_log(
    runtime=Path(__file__).parent.joinpath("logs/runtime.log"),
    error=Path(__file__).parent.joinpath("logs/error.log"),
    serialize=Path(__file__).parent.joinpath("logs/serialize.log"),
)
settings = Settings()

cache = Cache(directory=settings.DISKCACHE_FOLDER)

sj = json.dumps(settings.model_dump(mode="json"), indent=2, ensure_ascii=False)
logger.info(f"Load settings - {sj}")

try:
    tokenizer, model = load_model(settings.MODEL_NAME, settings.DEVICE)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    sys.exit()


class GotMode(str, Enum):
    plain_texts_ocr = "plain texts OCR"
    format_texts_ocr = "format texts OCR"
    plain_multi_crop_ocr = "plain multi-crop OCR"
    format_multi_crop_ocr = "format multi-crop OCR"
    plain_fine_grained_ocr = "plain fine-grained OCR"
    format_fine_grained_ocr = "format fine-grained OCR"


def disk_cache(func):
    """Decorator to implement disk caching for the run_got function."""

    @wraps(func)
    def wrapper(
        image: str,
        got_mode: str,
        fine_grained_mode: str | None = None,
        ocr_color: str | None = None,
        ocr_box: str | None = None,
    ) -> Tuple[str, Optional[str]]:
        # Generate a unique key based on the function arguments and image content
        image_hash = get_file_hash(image)
        cache_key = f"{image_hash}:{got_mode}:{fine_grained_mode}:{ocr_color}:{ocr_box}"

        # Check if the result is already in the cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # If not in cache, run the original function
        result = func(image, got_mode, fine_grained_mode, ocr_color, ocr_box)

        # Store the result in the cache
        cache.set(cache_key, result)

        return result

    return wrapper


@log_execution_time
@disk_cache
def run_got(
    image: str,
    got_mode: str,
    fine_grained_mode: str | None = None,
    ocr_color: str | None = None,
    ocr_box: str | None = None,
) -> Tuple[str, Optional[str]]:
    """Run the GOT model and return the results.

    Args:
        image (str): Path to the image.
        got_mode (str): Mode selection.
        fine_grained_mode (str, optional): Fine-grained mode. Defaults to None.
        ocr_color (str, optional): OCR color. Defaults to None.
        ocr_box (str, optional): OCR box. Defaults to None.

    Returns:
        Tuple[str, Optional[str]]: Result latex code and optional rendered HTML content.
    """
    logger.debug(f"run got - mode={got_mode} {ocr_color=} {ocr_box=}")

    unique_id = str(uuid.uuid4())
    image_path = os.path.join(settings.UPLOAD_FOLDER, f"{unique_id}.png")
    result_path = os.path.join(settings.RESULTS_FOLDER, f"{unique_id}.html")

    try:
        shutil.copy(image, image_path)

        if got_mode == GotMode.plain_texts_ocr:
            res = model.chat(tokenizer, image_path, ocr_type="ocr")
            return res, None
        elif got_mode == GotMode.format_texts_ocr:
            res = model.chat(
                tokenizer, image_path, ocr_type="format", render=True, save_render_file=result_path
            )
        elif got_mode == GotMode.plain_multi_crop_ocr:
            res = model.chat_crop(tokenizer, image_path, ocr_type="ocr")
            return res, None
        elif got_mode == GotMode.format_multi_crop_ocr:
            res = model.chat_crop(
                tokenizer, image_path, ocr_type="format", render=True, save_render_file=result_path
            )
        elif got_mode == GotMode.plain_fine_grained_ocr:
            res = model.chat(
                tokenizer, image_path, ocr_type="ocr", ocr_box=ocr_box, ocr_color=ocr_color
            )
            return res, None
        elif got_mode == GotMode.format_fine_grained_ocr:
            res = model.chat(
                tokenizer,
                image_path,
                ocr_type="format",
                ocr_box=ocr_box,
                ocr_color=ocr_color,
                render=True,
                save_render_file=result_path,
            )
        else:
            raise ValueError(f"Invalid mode: {got_mode}")

        # res_markdown = f"$$ {res} $$"
        res_markdown = res

        if "format" in got_mode and os.path.exists(result_path):
            with open(result_path, "r") as f:
                html_content = f.read()
            encoded_html = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
            iframe_src = f"data:text/html;base64,{encoded_html}"
            iframe = f'<iframe src="{iframe_src}" width="100%" height="600px"></iframe>'
            download_link = f'<a href="data:text/html;base64,{encoded_html}" download="result_{unique_id}.html">Download Full Result</a>'
            return res_markdown, f"{download_link}<br>{iframe}"
        else:
            return res_markdown, None
    except Exception as err:
        logger.exception(err)
        return f"Error: {str(err)}", None


def task_update(task: str) -> list:
    """Update UI components based on the selected task."""
    if "fine-grained" in task:
        return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]
    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]


def fine_grained_update(task: str) -> list:
    """Update UI components for fine-grained OCR options."""
    if task == "box":
        return [gr.update(visible=False, value=""), gr.update(visible=True)]
    if task == "color":
        return [gr.update(visible=True), gr.update(visible=False, value="")]


def cleanup_old_files() -> None:
    """Remove files older than FILE_EXPIRATION_TIME."""
    current_time = time.time()
    for folder_ in [settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER]:
        for path in Path(folder_).glob("**/*"):
            try:
                if current_time - path.stat().st_mtime > settings.FILE_EXPIRATION_TIME:
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
            except Exception as e:
                logger.error(f"Error while cleaning up {path}: {str(e)}")


title_html = """
<h2> <span class="gradient-text" id="text">General OCR Theory</span><span class="plain-text">: Towards OCR-2.0 via a Unified End-to-end Model</span></h2>
<a href="https://huggingface.co/ucaslcl/GOT-OCR2_0">[😊 Hugging Face]</a> 
<a href="https://arxiv.org/abs/2409.01704">[📜 Paper]</a> 
<a href="https://github.com/Ucas-HaoranWei/GOT-OCR2.0/">[🌟 GitHub]</a> 
"""
title_markdown = """
"🔥🔥🔥This is the official online demo of GOT-OCR-2.0 model!!!"
        
### Demo Guidelines
You need to upload your image below and choose one mode of GOT, then click "Submit" to run GOT model. More characters will result in longer wait times.
- **plain texts OCR & format texts OCR**: The two modes are for the image-level OCR.
- **plain multi-crop OCR & format multi-crop OCR**: For images with more complex content, you can achieve higher-quality results with these modes.
- **plain fine-grained OCR & format fine-grained OCR**: In these modes, you can specify fine-grained regions on the input image for more flexible OCR. Fine-grained regions can be coordinates of the box, red color, blue color, or green color.
"""


def build_got_server():
    with gr.Blocks() as demo:
        gr.HTML(title_html)
        gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="upload your image")
                task_dropdown = gr.Dropdown(
                    choices=[
                        GotMode.plain_texts_ocr.value,
                        GotMode.format_texts_ocr.value,
                        GotMode.plain_multi_crop_ocr.value,
                        GotMode.format_multi_crop_ocr.value,
                        GotMode.plain_fine_grained_ocr.value,
                        GotMode.format_fine_grained_ocr.value,
                    ],
                    label="Choose one mode of GOT",
                    value=GotMode.plain_texts_ocr.value,
                    allow_custom_value=True,
                )
                fine_grained_dropdown = gr.Dropdown(
                    choices=["box", "color"],
                    label="fine-grained type",
                    visible=False,
                    allow_custom_value=True,
                )
                color_dropdown = gr.Dropdown(
                    choices=["red", "green", "blue"],
                    label="color list",
                    visible=False,
                    allow_custom_value=True,
                )
                box_input = gr.Textbox(
                    label="input box: [x1,y1,x2,y2]",
                    placeholder="e.g., [0,0,100,100]",
                    visible=False,
                )
                submit_button = gr.Button("Submit")

            with gr.Column():
                ocr_result = gr.Textbox(label="GOT output")

        with gr.Column():
            gr.Markdown(
                "**If you choose the mode with format, the mathpix result will be automatically rendered as follows:**"
            )
            html_result = gr.HTML(label="rendered html", show_label=True)

        task_dropdown.change(
            task_update,
            inputs=[task_dropdown],
            outputs=[fine_grained_dropdown, color_dropdown, box_input],
            show_api=False,
        )
        fine_grained_dropdown.change(
            fine_grained_update,
            inputs=[fine_grained_dropdown],
            outputs=[color_dropdown, box_input],
            show_api=False,
        )

        submit_button.click(
            run_got,
            inputs=[image_input, task_dropdown, fine_grained_dropdown, color_dropdown, box_input],
            outputs=[ocr_result, html_result],
        )
    return demo


if __name__ == "__main__":
    cleanup_old_files()
    logger.success(f"Running on local URL:  http://{settings.SERVER_HOST}:{settings.SERVER_PORT}")
    got_server = build_got_server()
    got_server.launch(
        share=True,
        server_name="0.0.0.0",
        show_error=True,
        debug=True,
        server_port=settings.SERVER_PORT,
    )

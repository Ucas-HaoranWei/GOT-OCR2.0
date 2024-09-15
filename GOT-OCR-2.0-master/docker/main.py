"""
References:
    https://huggingface.co/ucaslcl/GOT-OCR2_0
    https://huggingface.co/spaces/ucaslcl/GOT_online/blob/main/app.py
    https://github.com/Ucas-HaoranWei/GOT-OCR2.0
"""

import asyncio
import base64
import io
import json
import os
import shutil
import sys
import time
import uuid
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Tuple, Optional

import gradio as gr
from PIL import Image
from huggingface_hub import snapshot_download
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import AutoModel, AutoTokenizer


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    RUNTIME_ENV: str = "dev"
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 7860

    MODEL_NAME: str = "ucaslcl/GOT-OCR2_0"
    HF_HOME: str = "~/.cache/huggingface/"

    UPLOAD_FOLDER: str = "uploads"
    RESULTS_FOLDER: str = "results"
    FILE_EXPIRATION_TIME: int = 3600


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
            return result

    return wrapper


# == Logger Init == #
init_log(
    runtime=Path(__file__).parent.joinpath("logs/runtime.log"),
    error=Path(__file__).parent.joinpath("logs/error.log"),
    serialize=Path(__file__).parent.joinpath("logs/serialize.log"),
)
settings = Settings()
sj = json.dumps(settings.model_dump(mode="json"), indent=2, ensure_ascii=False)
logger.info(f"Load settings - {sj}")

# == Model Init == #
logger.debug(f"Loading snapshot - {settings.MODEL_NAME}")
model_cache = snapshot_download(settings.MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(model_cache, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_cache,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="cuda",
    use_safetensors=True,
)
model = model.eval().cuda()
logger.success(f"Model cache path: {model_cache}")

# Ensure necessary folders exist
for folder in [settings.UPLOAD_FOLDER, settings.RESULTS_FOLDER]:
    Path(folder).mkdir(parents=True, exist_ok=True)


class GotMode(str, Enum):
    plain_texts_ocr = "plain texts OCR"
    format_texts_ocr = "format texts OCR"
    plain_multi_crop_ocr = "plain multi-crop OCR"
    format_multi_crop_ocr = "format multi-crop OCR"
    plain_fine_grained_ocr = "plain fine-grained OCR"
    format_fine_grained_ocr = "format fine-grained OCR"


def image_to_base64(image: Image) -> str:
    """Convert an image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


@log_execution_time
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
        Tuple[str, Optional[str]]: Result text and optional HTML link.
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
    except Exception as e:
        return f"Error: {str(e)}", None
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


def task_update(task: str) -> list:
    """Update UI components based on the selected task."""
    if "fine-grained" in task:
        return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]
    else:
        return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]


def fine_grained_update(task: str) -> list:
    """Update UI components for fine-grained OCR options."""
    if task == "box":
        return [gr.update(visible=False, value=""), gr.update(visible=True)]
    elif task == "color":
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

with gr.Blocks() as demo:
    gr.HTML(title_html)
    gr.Markdown(
        """
    "🔥🔥🔥This is the official online demo of GOT-OCR-2.0 model!!!"
    
    ### Demo Guidelines
    You need to upload your image below and choose one mode of GOT, then click "Submit" to run GOT model. More characters will result in longer wait times.
    - **plain texts OCR & format texts OCR**: The two modes are for the image-level OCR.
    - **plain multi-crop OCR & format multi-crop OCR**: For images with more complex content, you can achieve higher-quality results with these modes.
    - **plain fine-grained OCR & format fine-grained OCR**: In these modes, you can specify fine-grained regions on the input image for more flexible OCR. Fine-grained regions can be coordinates of the box, red color, blue color, or green color.
    """
    )

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
                label="input box: [x1,y1,x2,y2]", placeholder="e.g., [0,0,100,100]", visible=False
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
    )
    fine_grained_dropdown.change(
        fine_grained_update, inputs=[fine_grained_dropdown], outputs=[color_dropdown, box_input]
    )

    submit_button.click(
        run_got,
        inputs=[image_input, task_dropdown, fine_grained_dropdown, color_dropdown, box_input],
        outputs=[ocr_result, html_result],
    )

if __name__ == "__main__":
    cleanup_old_files()
    logger.success(f"Running on local URL:  http://{settings.SERVER_HOST}:{settings.SERVER_PORT}")
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        show_error=True,
        debug=True,
        server_port=settings.SERVER_PORT,
    )

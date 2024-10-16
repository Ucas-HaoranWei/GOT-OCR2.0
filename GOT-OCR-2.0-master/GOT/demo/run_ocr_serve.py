import argparse
import io
import os

import torch
from GOT.demo.process_results import punctuation_dict
from GOT.model import *
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import KeywordsStoppingCriteria
from GOT.utils.utils import disable_torch_init
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from starlette import status
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from transformers import AutoTokenizer
from transformers import TextStreamer

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

translation_table = str.maketrans(punctuation_dict)

parser = argparse.ArgumentParser()

args = argparse.Namespace()
args.model_name = "facebook/opt-350m"
args.type = "format"
args.box = ''
args.color = ''

disable_torch_init()
model_name = os.path.expanduser(args.model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()

model.to(device='cuda', dtype=torch.bfloat16)

# TODO vary old codes, NEED del
image_processor = BlipImageEvalProcessor(image_size=1024)

image_processor_high = BlipImageEvalProcessor(image_size=1024)

use_im_start_end = True

image_token_len = 256


def eval_model(image):
    # Model

    # image = load_image(args.image_file)

    w, h = image.size
    # print(image.size)

    if args.type == 'format':
        qs = 'OCR with format: '
    else:
        qs = 'OCR: '

    if args.box:
        bbox = eval(args.box)
        if len(bbox) == 2:
            bbox[0] = int(bbox[0] / w * 1000)
            bbox[1] = int(bbox[1] / h * 1000)
        if len(bbox) == 4:
            bbox[0] = int(bbox[0] / w * 1000)
            bbox[1] = int(bbox[1] / h * 1000)
            bbox[2] = int(bbox[2] / w * 1000)
            bbox[3] = int(bbox[3] / h * 1000)
        if args.type == 'format':
            qs = str(bbox) + ' ' + 'OCR with format: '
        else:
            qs = str(bbox) + ' ' + 'OCR: '

    if args.color:
        if args.type == 'format':
            qs = '[' + args.color + ']' + ' ' + 'OCR with format: '
        else:
            qs = '[' + args.color + ']' + ' ' + 'OCR: '

    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv_mode = "mpt"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print('=' * 100)
    print(prompt)
    print('=' * 100)

    inputs = tokenizer([prompt])

    # vary old codes, no use
    image_1 = image.copy()
    image_tensor = image_processor(image)

    image_tensor_1 = image_processor_high(image_1)

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print('开始推理')
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=20,
            streamer=streamer,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria]
        )
        print('*' * 100)
        print(output_ids)
        print('*' * 100)

    print('============== OCR 结果 ===============')
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs


app = FastAPI()


def add_cors_middleware(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


add_cors_middleware(app)


@app.post("/inference")
@app.options("/inference")
async def upload_image(file: UploadFile = File(...)):
    file_content = await file.read()
    image = Image.open(io.BytesIO(file_content)).convert('RGB')
    result = eval_model(image)
    return Response(
        status_code=status.HTTP_200_OK, media_type="text/plain",
        content=result
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app="run_ocr:app", host="0.0.0.0", port=6006, workers=1)

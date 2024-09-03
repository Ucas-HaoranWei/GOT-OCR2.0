import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

from tqdm import tqdm
from PIL import Image
import json
import os
import requests
from PIL import Image
from io import BytesIO
import math

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from GOT.model import *
from GOT.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
from GOT.model.plug.blip_process import BlipImageEvalProcessor

from transformers import TextStreamer
from GOT.model.plug.transforms import train_transform, test_transform
import re
from GOT.demo.process_results import punctuation_dict, svg_to_html

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


import string
 
translation_table = str.maketrans(punctuation_dict)


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=1024, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    # print(target_ratios)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # print(target_aspect_ratio)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # print(blocks)

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



output_list = []

def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()


    # vary old codes, no use
    image_processor = BlipImageEvalProcessor(image_size=1024)


    # image_processor_high = BlipImageEvalProcessor(image_size=1280)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)
    use_im_start_end = True



    # image_token_len = 400
    image_token_len = 256
    gts_path = args.gtfile_path
    gts = json.load(open(gts_path))

    # gts = gts[0]


    print("Generate Results......")


    if "OCR" in args.datatype:
            gts = get_chunk(gts, args.num_chunks, args.chunk_idx)


    for ann in tqdm(gts):
        output_json = {}
        
        if "OCR" in args.datatype:
            qs = ann["conversations"][0]["value"]
        else:
            qs = ann["question"]
            # ans = ann["answers"][0]
        
        qs2 = qs
        image_file = ann["image"] 
        if 'Text' in args.datatype:
            image_file = image_file + '.jpg'
        if "VQAv2" in args.datatype:
            image_file = 'COCO_' + 'val2014' + '_'+ str(image_file).zfill(12) + '.jpg'
        if "Cap" in args.datatype:
            image_file = 'COCO_' + 'val2014' + '_'+ str(image_file).zfill(12) + '.jpg'

        image_file_path = os.path.join(args.image_path, image_file)
        # print(image_file_path)
        # exit()

        # qs = args.query
        # if mm_use_im_start_end:



        multi_crop = False
        if multi_crop:
            image_list = []
            # qs =  DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN  + '\n' +  'OCR with format upon the patch reference: '
            img = load_image(image_file_path)
            sub_images = dynamic_preprocess(img)
            ll = len(sub_images)
            for p in sub_images:
                image = p
                image_1 = image.copy()
                # vary old code, NO USE
                image_tensor = image_processor_high(image_1)

                # image_tensor_1 = image_processor_high.preprocess(image_1, return_tensors='pt')['pixel_values'][0]

                image_tensor_1 = image_processor_high(image_1)

                image_list.append(image_tensor_1)

                # print(image_tensor_1.shape)

            image_list = torch.stack(image_list)

        else:
            ll = 1
            image = load_image(image_file_path)
            image_1 = image.copy()
            # image_1 = image_1.resize((1024, 1024))

            # vary old code, NO USE
            image_tensor = image_processor_high(image_1)

            image_tensor_1 = image_processor_high(image_1)
            # image_tensor_1 = torch.zeros(3, 1024, 1024)


        qs =  DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len*ll + DEFAULT_IM_END_TOKEN  + '\n' +  'OCR with format: '

        

        conv_mode = "mpt"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        if multi_crop:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids,
                    images=[(image_list.half().cuda(), image_list.half().cuda())],
                    do_sample=False,
                    num_beams = 1,
                    # temperature=0.2,
                    # no_repeat_ngram_size = 20,
                    # streamer=streamer,
                    max_new_tokens=4096,
                    stopping_criteria=[stopping_criteria]
                    )
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids,
                    images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                    do_sample=False,
                    num_beams = 1,
                    # temperature=0.2,
                    no_repeat_ngram_size = 20,
                    # encoder_repetition_penalty = 1.2,
                    # penalty_alpha=0.2,
                    # top_k=3,
                    max_new_tokens=4096,
                    stopping_criteria=[stopping_criteria]
                    )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # outputs = outputs.strip()[:-1]
        if "Cap" in args.datatype:
            # output_json['image'] = ann["image"]
            output_json['image_id'] = ann["id"]
            output_json["caption"] = outputs
        else:
            # output_json['questionId'] = qs_id
            # output_json['question_id'] = qs_id
            output_json['image'] = ann["image"]
            output_json['question'] = qs 
            output_json['label'] = ann["conversations"][1]["value"]
            output_json['answer'] = outputs
        output_list.append(output_json)

    filename = args.out_path + "/results_" + str(args.chunk_idx) + ".json"
    with open(filename, 'w', encoding="utf-8") as file_obj:
        json.dump(output_list, file_obj, ensure_ascii=False, indent=1)
        # print(outputs)
    # print("Evaluate Results... ")
    # doc_text_eval(gts_path, filename, args.datatype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--gtfile_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--datatype", type=str, required=True)  # Text or Doc
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    print(args)
    eval_model(args)

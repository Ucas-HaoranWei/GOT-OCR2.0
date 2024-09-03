import json
import argparse
import nltk
from nltk.metrics import precision, recall, f_measure
import numpy as np
import jieba
# import megfile as mf
import pickle
import pandas as pd
import re
from nltk.translate import meteor_score

parser = argparse.ArgumentParser()

parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--gt_path", type=str, required=True)
parser.add_argument("--datatype", type=str, required=True)
args = parser.parse_args()

def preprocess(text, predict_root_):
    if 'InternVL' in predict_root_:
        text = text.split("All words in the image:\n")[1]
        text = text.split("[UNUSED_TOKEN_145]")[0]
    return text

def contain_chinese_string(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))

def nougat_per_metrics(predict_root_, pred, gt, minlen=1):

    metrics = {}

    if len(pred) < minlen or len(gt) < minlen:
        return metrics


    reference = list(gt)
    hypothesis = list(pred)

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)

    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)
    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))

    return metrics

def doc_text_eval(gt_root_, predict_root_, datatype):

   

    predicts = json.load(open(predict_root_, encoding='utf-8'))
    
    result = []
    for ann in predicts:
        try:
            ans = nougat_per_metrics(predict_root_, ann["label"], ann["answer"])
            if len(ans) == 0:
                continue
            result.append(ans)
        except:
            assert False, print("ERROR!!! Check yout output!!!")
    
    mean_dict = {}

    mean_dict["eval question num"] = len(result)
    for k, v in result[0].items():
        mean_dict[k] = 0
    
    for each in result:
        for k, v in each.items():
            mean_dict[k] += v

    for k, v in mean_dict.items():
        if k == "eval question num":
            continue
        mean_dict[k] /= len(result)
    print(json.dumps(mean_dict, indent=4))


doc_text_eval(args.gt_path, args.out_path + "/results_final.json", args.datatype)
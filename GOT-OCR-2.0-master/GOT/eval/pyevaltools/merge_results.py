import os
import json
import argparse

def merge_outputs(out_path):
    files = os.listdir(out_path)
    # print(files)
    alist = []
    for file in files:
        alist += json.load(open(os.path.join(out_path, file), encoding='utf-8'))
    # print(len(alist))

    filename = out_path + "/results_final" + ".json"
    with open(filename, 'w', encoding="utf-8") as file_obj:
        json.dump(alist, file_obj, ensure_ascii=False, indent=1)

parser = argparse.ArgumentParser()
parser.add_argument("--out_path", type=str, required=True)
args = parser.parse_args()

merge_outputs(args.out_path)
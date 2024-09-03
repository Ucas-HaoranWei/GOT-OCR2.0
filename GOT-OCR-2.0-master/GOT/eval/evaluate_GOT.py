import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
parser.add_argument("--gtfile_path", type=str, required=True)
parser.add_argument("--image_path", type=str, required=True)
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--num-chunks", type=int, default=1)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--datatype", type=str, required=True)  # Text\Doc\VQAv2\Cap
# parser.add_argument("--eval", type=str, required=True)
args = parser.parse_args()

os.system("python3 -m GOT.eval.multi_hardware_eval_GOT" + " "
          + "--model-name" + " " + args.model_name + " "
          + "--gtfile_path" + " " + args.gtfile_path + " "
          + "--image_path" + " " + args.image_path + " "
          + "--out_path" + " " + args.out_path + " "
          + "--num-chunks" + " " + str(args.num_chunks) + " "
          + "--temperature" + " " + str(args.temperature) + " "
          + "--datatype" + " " + args.datatype
          )

print("Evaluating.....")
os.system("python3 -m GOT.eval.pyevaltools.merge_results" + " "
          + "--out_path" + " " + args.out_path)


# if args.datatype == "OCR":


a_type = 'plain'  # 'palin'; 'format'; 'scene'

if a_type == 'plain':
    os.system("python3 -m GOT.eval.pyevaltools.eval_ocr" + " "
                + "--out_path" + " " + args.out_path + " "
                + "--gt_path" + " " + args.gtfile_path + " "
                + "--datatype" + " " + args.datatype
                )
if a_type == 'format':
    os.system("python3 -m GOT.eval.pyevaltools.eval_ocr_format" + " "
            + "--out_path" + " " + args.out_path + " "
            + "--gt_path" + " " + args.gtfile_path + " "
            + "--datatype" + " " + args.datatype
            )
if a_type == 'scene':
    os.system("python3 -m GOT.eval.pyevaltools.eval_ocr_scene" + " "
        + "--out_path" + " " + args.out_path + " "
        + "--gt_path" + " " + args.gtfile_path + " "
        + "--datatype" + " " + args.datatype
        )
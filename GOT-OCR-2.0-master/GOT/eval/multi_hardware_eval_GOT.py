import os
import argparse
from multiprocessing import Pool
# from GOT.eval.merge_results import merge_outputs
# from GOT.eval.doctextVQA import doc_text_eval


def run_eval(chunk_id, model_name, gtfile_path, image_path, out_path, num_chunks, datatype, temperature):
    os.system("CUDA_VISIBLE_DEVICES=" + str(chunk_id) + " "
              + "python3 -m GOT.eval.eval_GOT_ocr" + " "
              + "--model-name" + " " + model_name + " "
              + "--gtfile_path" +  " " + gtfile_path + " "
              + "--image_path" + " " + image_path + " "
              + "--out_path" +  " " + out_path + " "
              + "--num-chunks" + " " +  str(num_chunks) + " "
              + "--chunk-idx" + " " +  str(chunk_id) + " "
              + "--temperature" + " " +  str(temperature) + " "
              + "--datatype" + " " + datatype
              )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--gtfile_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--datatype", type=str, required=True)  # Text or Doc
    # parser.add_argument("--eval", type=str, required=True)
    args = parser.parse_args()

    num_chunks = args.num_chunks
    
    if os.path.exists(args.out_path) == False:
        os.makedirs(args.out_path)


    with Pool(num_chunks) as p:
        for i in range(num_chunks):
            chunk_id = i
            p.apply_async(run_eval, (chunk_id, args.model_name, args.gtfile_path,
                                     args.image_path, args.out_path, num_chunks, args.datatype, args.temperature))
        p.close()
        p.join()


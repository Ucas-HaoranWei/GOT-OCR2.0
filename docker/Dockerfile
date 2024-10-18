FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y wget libgl1-mesa-glx libglib2.0-0

# https://huggingface.co/spaces/ucaslcl/GOT_online/blob/main/requirements.txt
# https://github.com/Ucas-HaoranWei/GOT-OCR2.0/blob/main/GOT-OCR-2.0-master/pyproject.toml#L15
RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    modelscope \
    transformers \
    markdown2[all]  \
    sentencepiece \
    tokenizers>=0.15.2 \
    wandb \
    shortuuid \
    httpx \
    deepspeed==0.12.3 \
    peft==0.4.0 \
    albumentations \
    opencv-python \
    tiktoken==0.6.0 \
    accelerate==0.28.0 \
    bitsandbytes==0.41.0 \
    scikit-learn==1.2.2 \
    sentencepiece==0.1.99 \
    einops==0.6.1 \
    einops-exts==0.0.4  \
    timm==0.6.13 \
    gradio \
    natsort \
    python-dotenv \
    verovio \
    numpy~=1.26.4 \
    megfile==3.1.2 \
    pydantic-settings \
    loguru \
    diskcache

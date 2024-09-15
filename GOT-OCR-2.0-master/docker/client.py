from enum import Enum
from typing import Tuple, Optional

from gradio_client import Client, handle_file
from loguru import logger


class GotMode(str, Enum):
    plain_texts_ocr = "plain texts OCR"
    format_texts_ocr = "format texts OCR"
    plain_multi_crop_ocr = "plain multi-crop OCR"
    format_multi_crop_ocr = "format multi-crop OCR"
    plain_fine_grained_ocr = "plain fine-grained OCR"
    format_fine_grained_ocr = "format fine-grained OCR"


def demo(
    filepath_or_url: str,
    got_mode: GotMode = GotMode.plain_texts_ocr,
    fine_grained_mode: str = None,
    ocr_color: str = None,
    ocr_box: str = None,
    *,
    base_url: str = "http://127.0.0.1:7860/"
) -> Tuple[str, Optional[str]]:
    client = Client(src=base_url)
    result = client.predict(
        image=handle_file(filepath_or_url),
        got_mode=got_mode,
        fine_grained_mode=fine_grained_mode,
        ocr_color=ocr_color,
        ocr_box=ocr_box,
        api_name="/run_got",
    )
    logger.debug(result)

    # [0] type:str got output
    # [1] type:str rendered html
    return result


if __name__ == "__main__":
    demo(
        filepath_or_url="https://raw.githubusercontent.com/Ucas-HaoranWei/GOT-OCR2.0/main/assets/weichat2.jpg",
        got_mode=GotMode.plain_texts_ocr,
        base_url="http://127.0.0.1:7860/",  # modify it to your own
    )

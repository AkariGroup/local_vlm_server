from abc import ABCMeta, abstractmethod
import cv2
from PIL import Image
from typing import List, Optional, Union
import numpy as np
import io
import base64

VLM_LIST = {
    "smolvlm": [
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        "HuggingFaceTB/SmolVLM-500M-Instruct",
    ],
    "moondream": [
        "vikhyatk/moondream2",
    ],
    "lfm2vl": [
        "LiquidAI/LFM2-VL-450M",
        "LiquidAI/LFM2-VL-1.6B",
    ],
    "heron": {
        "turing-motors/Heron-NVILA-Lite-1B",
        "turing-motors/Heron-NVILA-Lite-2B",
        "turing-motors/Heron-NVILA-Lite-15B",
    },
    "fastvlm": {
        "apple/FastVLM-0.5B",
        "apple/FastVLM-1.5B",
        "apple/FastVLM-7B",
    },
    "qwen": {
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
    },
    "internvl3": {
        "OpenGVLab/InternVL3-1B",
        "OpenGVLab/InternVL3-2B",
        "OpenGVLab/InternVL3-8B",
        "OpenGVLab/InternVL3-9B",
        "OpenGVLab/InternVL3-14B",
        "OpenGVLab/InternVL3-38B",
        "OpenGVLab/InternVL3-78B",
    }

}


def get_vlm_category(model_name) -> Optional[str]:
    """
    VLMのカテゴリを取得する
    Args:
        model_name (str): モデル名
    Returns:
        str: VLMのカテゴリ
    """
    for category, models in VLM_LIST.items():
        if model_name in models:
            return category
    return None


class Vlm(metaclass=ABCMeta):
    """
    VLMの基底クラス
    """
    def __init__(self) -> None:
        """
        VLMの初期化
        """
        self.DEFAULT_PROMPT = "Please describe this image."

    def convert_to_pil(self, image: Union[np.ndarray, str]) -> Image:
        """
        画像をPIL形式に変換する
        Args:
            image (np.ndarray or str): 画像
        Returns:
            Image: PIL形式の画像
        """
        converted_image = None
        try:
            if isinstance(image, str):
                # Convert base64 string to PIL Image
                image_data = base64.b64decode(image)
                converted_image = Image.open(io.BytesIO(image_data))
            elif isinstance(image, np.ndarray):
                converted_image = Image.fromarray(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                )
        except Exception as e:
            print(f"Error in convert_to_pil: {e}")
            raise ValueError("Invalid image format")
        return converted_image

    @abstractmethod
    def chat(
        self, image: Union[np.ndarray, List[np.ndarray], str, List[str]], prompt: str
    ) -> str:
        """
        VLMに画像とテキストを送信し、レスポンスを返す
        Args:
            image (Image): 画像
            text (str): テキスト
        Returns:
            str: レスポンス
        """
        ...

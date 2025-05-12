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
    "heron": {
        "turing-motors/Heron-NVILA-Lite-1B",
        "turing-motors/Heron-NVILA-Lite-2B",
        "turing-motors/Heron-NVILA-Lite-15B",
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

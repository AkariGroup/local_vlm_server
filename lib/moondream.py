import torch
import numpy as np
from typing import List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from .vlm import Vlm


class Moondream(Vlm):
    """
    Moondreamを使用するクラス
    """

    def __init__(
        self, model_name: str = "vikhyatk/moondream2", revision: str = "2025-03-27"
    ) -> None:
        """
        Moondreamの初期化

        Args:
            model_name (str): モデル名。デフォルトは
                "HuggingFaceTB/SmolVLM-256M-Instruct"。
        """
        super().__init__()
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            revision=revision,
            trust_remote_code=True,
            device_map={"": "cuda"} if torch.cuda.is_available() else None,
        )

    def chat(
        self,
        images: Union[np.ndarray, List[np.ndarray], str, List[str]],
        prompt: Optional[str] = None,
    ) -> str:
        """
        画像をVLMに送信し、レスポンスを取得する"
        Args:
            images (np.ndarray or str): 画像
                (base64エンコードされた文字列またはNumPy配列)
            prompt (str): プロンプト。デフォルトは
                "Please describe this image."。
        Returns:
            str: VLMからのレスポンス
        """
        if prompt is None:
            prompt = self.DEFAULT_PROMPT
        image = None
        if isinstance(images, list):
            if len(images) <1:
                raise ValueError(
                    "Empty list of images."
                )
            if len(images) == 1:
                image = self.convert_to_pil(images[0])
            else:
                raise ValueError(
                    "Moondream does not support multiple images. Please provide a single image."
                )
        else:
            image = self.convert_to_pil(images)
        answer = self.model.query(image, prompt)["answer"]
        return answer

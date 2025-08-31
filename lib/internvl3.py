import torch
import numpy as np
from typing import List, Optional, Union
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image
from .vlm import Vlm


class InrernVl3(Vlm):
    """
    InternVl3を使用するクラス
    """

    def __init__(self, model_name: str = "OpenGVLab/InternVL3-2B") -> None:
        """
        InternVl3の初期化

        Args:
            model_name (str): モデル名。デフォルトは
                "OpenGVLab/InternVL3-2B"。
        """
        super().__init__()
        self.model_name = model_name
        self.pipe = pipeline(
            self.model,
            backend_config=TurbomindEngineConfig(session_len=16384, tp=1),
            chat_template_config=ChatTemplateConfig(model_name="internvl2_5"),
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
            if len(images) < 1:
                raise ValueError("Empty list of images.")
            if len(images) == 1:
                image = self.convert_to_pil(images[0])
            else:
                raise ValueError(
                    "InternVL3 does not support multiple images. Please provide a single image."
                )
        else:
            image = self.convert_to_pil(images)
        result = self.pipe((prompt, image))
        return result.text

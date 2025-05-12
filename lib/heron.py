import copy
import torch
import numpy as np
from typing import List, Optional, Union
from transformers import AutoConfig, AutoModel, GenerationConfig
from .vlm import Vlm


class HeronNvilaLite(Vlm):
    """
    Heron-NVILA-Liteを使用するクラス
    """

    def __init__(
        self, model_name: str = "HuggingFaceTB/turing-motors/Heron-NVILA-Lite-1B"
    ) -> None:
        """
        Heron-NVILA-Liteの初期化

        Args:
            model_name (str): モデル名。デフォルトは
                "HuggingFaceTB/Heron-NVILA-Lite-1B"。
        """
        super().__init__()
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_config(
            config, trust_remote_code=True, device_map="auto"
        ).to(self.device)

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
        image_list = []
        if isinstance(images, list):
            for image in images:
                image_list.append(self.convert_to_pil(image))
        else:
            image_list.append(self.convert_to_pil(images))

        generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.5,
            "do_sample": True,
        }
        generation_config = GenerationConfig(**generation_config)
        messages = copy.deepcopy(image_list)
        messages.append(prompt)
        result = self.model.generate_content(
            messages, generation_config=generation_config
        )
        return result

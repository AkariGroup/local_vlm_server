import torch
import numpy as np
from typing import List, Optional, Union
from transformers import AutoProcessor, AutoModelForVision2Seq
from .vlm import Vlm


class SmolVlm(Vlm):
    """
    SmolVLMを使用するクラス
    """

    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct") -> None:
        """
        SmolVLMの初期化

        Args:
            model_name (str): モデル名。デフォルトは
                "HuggingFaceTB/SmolVLM-256M-Instruct"。
        """
        super().__init__()
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            _attn_implementation=(
                "flash_attention_2" if self.device == "cuda" else "eager"
            ),
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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=image_list, return_tensors="pt")
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        result = generated_texts[0]
        if "Assistant:" in result:
            result = result.split("Assistant:")[1].strip()
        return result

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Union
from .vlm import Vlm


class FastVlm(Vlm):
    """
    FastVLMを使用するクラス
    """

    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_name: str = "apple/FastVLM-1.5B") -> None:
        """
        FastVLMの初期化

        Args:
            model_name (str): モデル名。デフォルトは
                "apple/FastVLM-1.5B"。
        """
        super().__init__()
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
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
                    "FastVLM does not support multiple images. Please provide a single image."
                )
        else:
            image = self.convert_to_pil(images)
        messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
        rendered = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        pre, post = rendered.split("<image>", 1)

        # Tokenize the text *around* the image token (no extra specials!)
        pre_ids = self.tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = self.tok(
            post, return_tensors="pt", add_special_tokens=False
        ).input_ids

        # Splice in the IMAGE token id (-200) at the placeholder position
        img_tok = torch.tensor([[self.IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.model.device)
        attention_mask = torch.ones_like(input_ids, device=self.model.device)

        px = self.model.get_vision_tower().image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"]
        px = px.to(self.model.device, dtype=self.model.dtype)

        # Generate
        with torch.no_grad():
            out = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=px,
                max_new_tokens=128,
            )
        result = self.tok.decode(out[0], skip_special_tokens=True)
        return result

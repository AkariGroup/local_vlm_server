import torch
import numpy as np
from typing import List, Optional, Union
from transformers import AutoTokenizer, AutoModel
from PIL import Image
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # モデルとトークナイザーの初期化
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto"
        ).eval()

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
        
        # 画像の処理
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
        
        # InternVL3の標準的な使い方に従って実装
        # 画像タグを含むプロンプトを作成
        question = f"<image>\n{prompt}"
        
        # モデルのchatメソッドを使用（正しい引数形式で）
        generation_config = dict(
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
        )
        
        # InternVL3のchatメソッドを呼び出し（history付き）
        response, history = self.model.chat(
            self.tokenizer, 
            image,
            question,
            generation_config,
            history=None,
            return_history=True
        )
        
        return response
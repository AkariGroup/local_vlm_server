from typing import List, Optional, Union

import torch
from PIL import Image
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor

from .vlm import Vlm


class Qwen(Vlm):
    """Qwen2.5-VL instruct models wrapper."""

    def __init__(
        self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        device_map = "auto" if self.device.type == "cuda" else None
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()

    def _prepare_images(
        self, images: Union[np.ndarray, List[np.ndarray], str, List[str]]
    ) -> List[Image.Image]:
        if isinstance(images, list):
            if not images:
                raise ValueError("Empty list of images.")
            return [self.convert_to_pil(image) for image in images]
        return [self.convert_to_pil(images)]

    def chat(
        self,
        images: Union[np.ndarray, List[np.ndarray], str, List[str]],
        prompt: Optional[str] = None,
    ) -> str:
        if prompt is None:
            prompt = self.DEFAULT_PROMPT

        pil_images = self._prepare_images(images)

        # Build multimodal conversation turn for the processor.
        message_content = []
        for _ in pil_images:
            message_content.append({"type": "image"})
        message_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": message_content}]
        chat_template = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[chat_template],
            images=pil_images,
            return_tensors="pt",
        ).to(self.device)

        generation_kwargs = dict(max_new_tokens=512, do_sample=False, temperature=0.0)
        generated_ids = self.model.generate(
            **inputs,
            **generation_kwargs,
        )

        # Drop the input prompt tokens from the generated sequence before decoding.
        prompt_length = inputs["input_ids"].shape[-1]
        output_ids = generated_ids[:, prompt_length:]

        text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        return text.strip()

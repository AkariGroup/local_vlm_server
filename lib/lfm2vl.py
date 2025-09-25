import torch
import numpy as np
from typing import List, Optional, Union
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

from .vlm import Vlm


class Lfm2Vl(Vlm):
    """Wrapper for LiquidAI LFM2-VL models."""

    def __init__(self, model_name: str = "LiquidAI/LFM2-VL-450M") -> None:
        super().__init__()
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        device_map = "auto" if self.device.type == "cuda" else None

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
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

        message_content = []
        for pil_image in pil_images:
            message_content.append({"type": "image", "image": pil_image})
        message_content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": message_content}]

        model_inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

        target_device = getattr(self.model, "device", self.device)
        model_inputs = model_inputs.to(target_device)

        generation_kwargs = dict(max_new_tokens=512, do_sample=False, temperature=0.0)
        outputs = self.model.generate(
            **model_inputs,
            **generation_kwargs,
        )

        text = self.processor.batch_decode(
            outputs, skip_special_tokens=True
        )[0]

        if "Assistant:" in text:
            text = text.split("Assistant:", 1)[1]
        if "assistant" in text.lower() and "\n" in text:
            fragments = text.split("\n", 1)
            if fragments[0].lower().strip().startswith("assistant"):
                text = fragments[1]
        return text.strip()

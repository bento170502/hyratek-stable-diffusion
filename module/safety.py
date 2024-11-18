# safety.py
import torch
import numpy as np
from PIL import Image
from typing import Tuple, List
from transformers import AutoFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

class SafetyChecker:
    def __init__(self):
        safety_model_id = "CompVis/stable-diffusion-safety-checker"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        self.checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    def _load_replacement(self, x: np.ndarray) -> np.ndarray:
        try:
            hwc = x.shape
            y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
            y = (np.array(y)/255.0).astype(x.dtype)
            assert y.shape == x.shape
            return y
        except Exception:
            return x

    def check_images(self, images: np.ndarray) -> Tuple[np.ndarray, List[bool]]:
        """Check images for safety concerns."""
        if images.ndim == 3:
            images = images[None, ...]
            
        # Convert to PIL for feature extraction
        pil_images = [(images[i] * 255).round().astype("uint8") for i in range(images.shape[0])]
        pil_images = [Image.fromarray(img) for img in pil_images]
        
        # Run safety checker
        safety_checker_input = self.feature_extractor(pil_images, return_tensors="pt")
        checked_images, has_nsfw = self.checker(
            images=images,
            clip_input=safety_checker_input.pixel_values
        )

        # Replace unsafe images
        for i, is_nsfw in enumerate(has_nsfw):
            if is_nsfw:
                checked_images[i] = self._load_replacement(checked_images[i])
                
        return checked_images, has_nsfw

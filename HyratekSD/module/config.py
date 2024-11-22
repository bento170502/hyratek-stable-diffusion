from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    prompt: str
    height: int = 384
    width: int = 384
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: Optional[int] = None
    sampler: Literal["ddim", "plms", "dpm_solver"] = "ddim"
    safety_check: bool = True
    add_watermark: bool = True
    output_dir: str = "outputs"

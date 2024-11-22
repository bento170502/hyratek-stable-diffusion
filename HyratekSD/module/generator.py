# module/generator.py
import torch
from PIL import Image
from typing import List, Optional, Union
from omegaconf import OmegaConf
from torch import autocast
from contextlib import nullcontext
from dataclasses import dataclass
from .utils import save_images

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
    sampler: str = "ddim"
    safety_check: bool = True
    add_watermark: bool = False
    output_dir: str = "outputs"

class StableDiffusionGenerator:
    """Main interface for generating images with Stable Diffusion."""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "configs/stable-diffusion/v1-inference.yaml",
        device: Optional[str] = None
    ):
        """
        Initialize the generator.
        
        Args:
            model_path: Path to the model checkpoint
            config_path: Path to the model config file
            device: Device to use for inference ("cuda" or "cpu")
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(config_path, model_path)
        
    def _load_model(self, config_path: str, model_path: str):
        """Load the model from checkpoint."""
        from HyratekSD.ldm.util import instantiate_from_config
        
        print(f"Loading model from {model_path}")
        config = OmegaConf.load(config_path)
        
        # Load checkpoint
        pl_sd = torch.load(model_path, map_location="cpu")
        sd = pl_sd["state_dict"]
        
        # Initialize and load model
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        
        # Move to device and eval mode
        if self.device == "cuda":
            model.cuda()
        model.eval()
        
        return model

    def _get_sampler(self, sampler_type: str):
        """Get the appropriate sampler."""
        if sampler_type == "dpm_solver":
            from HyratekSD.ldm.models.diffusion.dpm_solver import DPMSolverSampler
            return DPMSolverSampler(self.model)
        elif sampler_type == "plms":
            from HyratekSD.ldm.models.diffusion.plms import PLMSSampler
            return PLMSSampler(self.model)
        else:  # default to ddim
            from HyratekSD.ldm.models.diffusion.ddim import DDIMSampler
            return DDIMSampler(self.model)

    def generate(
        self,
        config: Union[GenerationConfig, dict]
    ) -> List[Image.Image]:
        """
        Generate images based on the provided configuration.
        
        Args:
            config: Generation configuration
            
        Returns:
            List of generated images as PIL Images
        """
        if isinstance(config, dict):
            config = GenerationConfig(**config)
            
        # Set random seed if provided
        if config.seed is not None:
            torch.manual_seed(config.seed)
            
        # Setup sampler
        sampler = self._get_sampler(config.sampler)
        
        # Prepare for generation
        batch_size = 1  # Process one at a time to manage memory
        shape = [4, config.height // 8, config.width // 8]
        
        # Get conditioning
        uc = None
        if config.guidance_scale != 1.0:
            uc = self.model.get_learned_conditioning(batch_size * [""])
        c = self.model.get_learned_conditioning([config.prompt])
        
        # Generate images
        precision_scope = autocast if self.device == "cuda" else nullcontext
        generated_images = []
        
        with precision_scope("cuda"):
            with self.model.ema_scope():
                for _ in range(config.num_images):
                    # Generate latents
                    samples, _ = sampler.sample(
                        S=config.num_inference_steps,
                        conditioning=c,
                        batch_size=batch_size,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=config.guidance_scale,
                        unconditional_conditioning=uc,
                        eta=0.0
                    )
                    
                    # Decode to image space
                    x_samples = self.model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
                    
                    # Convert to PIL images
                    pil_images = []
                    for x_sample in x_samples:
                        x_sample = (x_sample * 255).round().astype("uint8")
                        pil_images.append(Image.fromarray(x_sample))
                    
                    generated_images.extend(pil_images)
                    
                    # Clear CUDA cache between generations
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
        
        # Save images if output directory is provided
        if config.output_dir:
            save_images(generated_images, config.output_dir, add_watermark=config.add_watermark)
        
        return generated_images
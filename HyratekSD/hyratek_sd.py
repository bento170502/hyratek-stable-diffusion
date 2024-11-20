# HyratekSD/hyratek_sd.py
from typing import List, Optional
from PIL import Image
from module.generator import StableDiffusionGenerator, GenerationConfig

class HyratekStableDiffusion:
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        height: int = 512,
        width: int = 512,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        sampler: str = "ddim",
        device: Optional[str] = None,
        enable_safety_checker: bool = True,
        enable_watermark: bool = False,
        output_dir: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to model checkpoint
            height: Default image height (default: 512)
            width: Default image width (default: 512)
            num_steps: Default number of inference steps (default: 50)
            guidance_scale: Default guidance scale (default: 7.5)
            sampler: Sampler to use - "ddim", "plms", or "dpm_solver" (default: "ddim")
            device: 'cuda' or 'cpu' (default: auto-detect)
            enable_safety_checker: Whether to check images for NSFW content
            enable_watermark: Whether to add watermarks to images
            output_dir: Default directory to save images (optional)
        """
        # Initialize generator
        self.generator = StableDiffusionGenerator(
            model_path=model_path,
            config_path=config_path,
            device=device
        )
        
        # Create base configuration
        self.config = GenerationConfig(
            prompt="",  # Will be updated in infer()
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            sampler=sampler,
            safety_check=enable_safety_checker,
            add_watermark=enable_watermark,
            output_dir=output_dir
        )

    def infer(
        self,
        prompt: str,
        num_images: int = 1,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Args:
            prompt: Text description
            num_images: Number of images to generate (default: 1)
            seed: Random seed for reproducibility
            
        Returns:
            List of generated images
        """
        # Update the config with the prompt and current parameters
        self.config.prompt = prompt
        self.config.num_images = num_images
        self.config.seed = seed
        
        # Generate images
        return self.generator.generate(self.config)

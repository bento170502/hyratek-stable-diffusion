# module/utils.py
import os
import cv2
import numpy as np
from PIL import Image
from typing import List
from imwatermark import WatermarkEncoder

def create_watermark() -> WatermarkEncoder:
    """Initialize watermark encoder."""
    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', "StableDiffusionV1".encode('utf-8'))
    return encoder

def apply_watermark(image: Image.Image, encoder: WatermarkEncoder) -> Image.Image:
    """Add invisible watermark to image."""
    if encoder is not None:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img = encoder.encode(img, 'dwtDct')
        image = Image.fromarray(img[:, :, ::-1])
    return image

def save_images(
    images: List[Image.Image],
    output_dir: str,
    base_count: int = 0,
    add_watermark: bool = True
) -> List[str]:
    """Save generated images to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    encoder = create_watermark() if add_watermark else None
    
    for i, image in enumerate(images):
        if add_watermark:
            image = apply_watermark(image, encoder)
            
        path = os.path.join(output_dir, f"{base_count + i:05d}.png")
        image.save(path)
        saved_paths.append(path)
        
    return saved_paths
# Description

# Install
```bash
docker-compose build
```
Download the Stable Diffusion model checkpoint and place it in the correct directory:

```bash
# Create model directory
mkdir -p models/ldm/stable-diffusion-v1/
# Place your model.ckpt file in this directory
# models/ldm/stable-diffusion-v1/model.ckpt
```
# How to run

* Python
```bash
from HyratekSD.hyratek_sd import HyratekSD
```
Initialize
```bash
sd = HyratekSD(
    model_path="models/ldm/stable-diffusion-v1/model.ckpt",
    height=512,
    width=512
)

# Generate images
images = sd.infer(
    prompt="a beautiful mountain landscape at sunset",
    num_images=1
)
```
* Docker
```bash
docker-compose run --rm hyratek-stable-diffusion python tests/test_generation.py
```
# References:

Model Weights Download the Stable Diffusion v1 weights from the CompVis organization on Hugging Face : https://huggingface.co/CompVis/stable-diffusion-v-1-1-original


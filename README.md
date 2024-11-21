# Description

# Install
*Docker
```bash
docker-compose build
```
*Conda
Environment Setup Create and activate the ldm environment:
```bash
conda env create -f environment.yaml
conda activate ldm
```
Create model directory, Link the weights in the specified directory:
```bash
mkdir -p models/ldm/stable-diffusion-v1/
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt
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
```bash
python test_generation.py
```
* Docker
```bash
docker-compose run --rm hyratek-stable-diffusion python tests/test_generation.py
```
# References:

Model Weights Download the Stable Diffusion v1 weights from the CompVis organization on Hugging Face : https://huggingface.co/CompVis/stable-diffusion-v-1-1-original


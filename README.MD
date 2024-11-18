Model Weights Download the Stable Diffusion v1 weights from the CompVis organization on Hugging Face : https://huggingface.co/CompVis/stable-diffusion-v-1-1-original

Link the weights in the specified directory:

mkdir -p models/ldm/stable-diffusion-v1/
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt
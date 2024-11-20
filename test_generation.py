from HyratekSD.hyratek_sd import HyratekStableDiffusion
import os

def main():

    rootDir = os.path.dirname(os.path.abspath(__file__))

    # Initialize the generator with custom settings
    sd = HyratekStableDiffusion(
        model_path=os.path.join(rootDir, "models/ldm/stable-diffusion-v1/model.ckpt"),
        config_path=os.path.join(rootDir, "configs/stable-diffusion/v1-inference.yaml"),
        height=512,
        width=512,
        num_steps=75,
        guidance_scale=8.5,
        output_dir="generated_images"
    )
    
    # Create output directory if it doesn't exist
    os.makedirs("generated_images", exist_ok=True)
    
    # Test single image generation
    print("Generating single image...")
    images = sd.infer(
        prompt="a king of lions is around animals",
        num_images=1,
        seed=42
    )
    print(f"Generated {len(images)} image(s)")
    
    # Test multiple images generation
    '''print("\nGenerating multiple images...")'''
    '''images = sd.infer(
        prompt="a futuristic cityscape with flying cars",
        num_images=4,
        seed=123
    )
    print(f"Generated {len(images)} image(s)")'''

if __name__ == "__main__":
    main()
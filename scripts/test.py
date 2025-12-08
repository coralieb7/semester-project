import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vlm.image_analyser import ImageAnalyzer
from config.model_config import ModelConfig
from src.vlm.universal_evolution_generator import UniversalEvolutionGenerator
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from src.sdxl.latent_slerp_interpolator import LatentSlerpInterpolator
from PIL import Image


device = 'cuda'
model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
image = Image.open("/home/eelab-fractal2/banuls/sdxl-adaptive-visuals/data/output/latent_slerp_sunflower/frames/keyframe_001.png")


img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)


# Test if VAE encode/decode works
print("\n=== Testing VAE ===")
test_image = image
interpolator = LatentSlerpInterpolator(
                vae=img2img_pipe.vae,
                device=ModelConfig.SDXL_DEVICE
            )
reconstructed = interpolator.test_encode_decode(test_image)
reconstructed.save("debug_reconstruction.png")
print("Check debug_reconstruction.png - should look like original")

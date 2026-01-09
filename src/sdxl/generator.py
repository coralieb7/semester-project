"""
SDXL Components
===============
FILE: src/sdxl/generator.py
"""

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
from typing import Optional, List
import gc
from pathlib import Path


class SDXLGenerator:
    """SDXL Generator with txt2img and img2img support"""
    
    def __init__(self, 
                 model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 device: str = "cuda",
                 enable_optimizations: bool = True,
                 load_txt_model: bool = False):
        """
        Initialize SDXL Generator
        
        Args:
            model_id: SDXL model ID
            device: Device to run on
            enable_optimizations: Enable memory optimizations
            load_txt_model: Use generated initial image
        """
        self.device = device
        self.model_id = model_id
        self.load_txt_model = load_txt_model
        
        print(f"Loading SDXL pipelines: {model_id}")
        
        # Load pipelines
        if self.load_txt_model:
            self.txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to(device)
        
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)
        
        if enable_optimizations:
            self._enable_optimizations()
        
        self.lora_loaded = False
        print("✓ SDXL Generator initialized")
    
    def _enable_optimizations(self):
        """Enable memory and speed optimizations"""
        if self.load_txt_model:
            self.txt2img_pipe.enable_attention_slicing()
        self.img2img_pipe.enable_attention_slicing()
        
        try:
            if self.load_txt_model:
                self.txt2img_pipe.enable_xformers_memory_efficient_attention()
            self.img2img_pipe.enable_xformers_memory_efficient_attention()
            print("  ✓ xformers enabled")
        except:
            print("  ℹ xformers not available")
    
    def load_lora(self, lora_path: str, adapter_name: str = "custom_lora"):
        """
        Load LoRA weights
        
        Args:
            lora_path: Path to LoRA weights directory
            adapter_name: Name for the LoRA adapter
        """
        print(f"Loading LoRA from: {lora_path}")
        if self.load_txt_model:
            self.txt2img_pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        self.img2img_pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        
        self.lora_loaded = True
        print(f"✓ LoRA '{adapter_name}' loaded")
    
    def unload_lora(self):
        """Unload LoRA weights"""
        if self.lora_loaded:
            if self.load_txt_model:
                self.txt2img_pipe.unload_lora_weights()
            self.img2img_pipe.unload_lora_weights()
            self.lora_loaded = False
            print("✓ LoRA unloaded")
    
    def generate_initial_image(self,
                              prompt: str,
                              negative_prompt: str = "blurry, low quality, distorted",
                              width: int = 1024,
                              height: int = 1024,
                              guidance_scale: float = 7.5,
                              num_inference_steps: int = 50,
                              seed: Optional[int] = None) -> Image.Image:
        """
        Generate initial image from text (txt2img)
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            guidance_scale: Guidance scale
            num_inference_steps: Number of steps
            seed: Random seed
            
        Returns:
            Generated PIL Image
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        print(f"Generating initial image (txt2img)...")
        print(f"  Prompt: {prompt[:80]}...")
        
        image = self.txt2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]
        
        return image
    
    def generate_keyframes_from_prompts(self,
                                       initial_image: Image.Image,
                                       prompts: List[str],
                                       negative_prompt: str = "no blurry, no low quality, no distorted, no text, no background, no distraction",
                                       strength: float = 0.25,
                                       guidance_scale: float = 15.0,
                                       num_inference_steps: int = 40,
                                       seed: Optional[int] = None,
                                       save_dir: Optional[str] = None) -> List[Image.Image]:
        """
        Generate keyframes using img2img with evolutionary prompts
        
        Args:
            initial_image: Starting image
            prompts: List of evolutionary prompts
            negative_prompt: Negative prompt
            strength: Transformation strength (0.0-1.0)
            guidance_scale: Guidance scale
            num_inference_steps: Inference steps
            seed: Random seed (will increment for each frame)
            save_dir: Optional directory to save keyframes
            
        Returns:
            List of generated keyframe images
        """
        print(f"\n{'='*70}")
        print(f"Generating {len(prompts)} keyframes")
        print(f"{'='*70}")
        print(f"Strength: {strength}, Guidance: {guidance_scale}, Steps: {num_inference_steps}")
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        keyframes = []
        current_image = initial_image
        
        # # Save initial image
        # if save_dir:
        #     initial_image.save(save_path / "keyframe_000.png")
        #     print(f"  Saved: keyframe_000.png (initial)")
        
        # Generate each keyframe
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Generating keyframe...")
            print(f"  Prompt: {prompt[:80]}...")
            
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed+i)
            
            # Generate next image
            current_image = self.img2img_pipe(
                prompt=prompt,
                image=current_image,
                negative_prompt=negative_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images[0]
            
            keyframes.append(current_image)
            
            # Save keyframe
            if save_dir:
                filename = f"keyframe_{i:03d}.png"
                current_image.save(save_path / filename)
                print(f"  Saved: {filename}")
            
            # Clear cache periodically
            if i % 3 == 0:
                self.clear_cache()
        
        print(f"\n✓ Generated {len(keyframes)} keyframes")
        return keyframes
    
    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

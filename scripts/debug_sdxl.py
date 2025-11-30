"""
FILE: scripts/debug_sdxl.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from PIL import Image
from src.vlm.prompt_generator import PromptGenerator
from src.sdxl.generator import SDXLGenerator
from src.sdxl.interpolator import FrameInterpolator
from src.sdxl.mapper import SignalMapper
from config.model_config import ModelConfig
from config.paths import Paths
import json
import torch
import os
import gc
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using VLM + SDXL"
    )
    
    # Input options
    parser.add_argument('--initial-image', type=str, required=True,
                       help='Path to initial/target image')
    
    # Generation parameters
    parser.add_argument('--path-prompts', type=str, required=True,
                       help='path to json file containing evolution prompts')
    parser.add_argument('--num-keyframes', type=int, default=None,
                       help='Number of keyframes to generate (default: same as prompts)')
    parser.add_argument('--interpolate', action='store_true',
                        help='Whether to interpolate between keyframes')
    parser.add_argument('--interpolations', type=int,
                       help='Interpolations between keyframes')
    
    # SDXL parameters
    parser.add_argument('--strength', type=float, default=0.45,
                       help='SDXL strength (0.0-1.0)')
    parser.add_argument('--guidance', type=float, default=15.0,
                       help='Guidance scale')
    parser.add_argument('--steps', type=int, default=50,
                       help='Inference steps')
    parser.add_argument('--width', type=int, default=1024,
                       help='Image width')
    parser.add_argument('--height', type=int, default=1024,
                       help='Image height')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Options
    parser.add_argument('--use-txt2img', action='store_true',
                       help='Start with txt2img instead of initial image')
    parser.add_argument('--lora', type=str, default=None,
                       help='Path to LoRA weights')
    parser.add_argument('--use-rag', action='store_true',
                       help='Use RAG for guideline enhancement')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory name')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SDXL Debug Tool")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output or f"debug_generation_{timestamp}"
    output_dir = Paths.OUTPUT_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    
    vlm = PromptGenerator(
        model_id=ModelConfig.VLM_MODEL_ID,
        device=ModelConfig.VLM_DEVICE
    )
    
    evolutionary_prompts = vlm.load_prompts(str(args.path_prompts))
    vlm.display_prompts(evolutionary_prompts)
    
    del vlm
    gc.collect()
    torch.cuda.empty_cache()
    sdxl = SDXLGenerator(
        model_id=ModelConfig.SDXL_MODEL_ID,
        device=ModelConfig.SDXL_DEVICE,
        load_txt_model=args.use_txt2img
    )
    
    # Load LoRA if specified
    if args.lora:
        sdxl.load_lora(args.lora)
    
    # Determine which prompts to use for keyframes
    num_keyframes = args.num_keyframes or len(evolutionary_prompts)
    
    if num_keyframes < len(evolutionary_prompts):
        # Select subset of prompts
        step = len(evolutionary_prompts) / num_keyframes
        selected_indices = [int(i * step) for i in range(num_keyframes)]
        selected_prompts = [evolutionary_prompts[i]['prompt'] for i in selected_indices]
    else:
        selected_prompts = [p['prompt'] for p in evolutionary_prompts]
    
    # Get initial image
    if args.use_txt2img:
        print("\nGenerating initial image with txt2img...")
        initial_image = sdxl.generate_initial_image(
            prompt=selected_prompts[0],
            width=args.width,
            height=args.height,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            seed=args.seed
        )
        selected_prompts = selected_prompts[1:]  # Skip first prompt
    else:
        print(f"\nLoading initial image: {args.initial_image}")
        initial_image = Image.open(args.initial_image).convert('RGB')
        initial_image = initial_image.resize((args.width, args.height), Image.LANCZOS)
    
    # Save initial image
    initial_image.save(frames_dir / "initial.png")
    
    # Generate keyframes
    keyframes = sdxl.generate_keyframes_from_prompts(
        initial_image=initial_image,
        prompts=selected_prompts,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        seed=args.seed,
        save_dir=str(frames_dir)
    )
    
    if args.interpolate:
        # Step 3: Interpolate frames
        print("\n" + "="*70)
        print("STEP 3: Interpolating frames")
        print("="*70)
        
        interpolator = FrameInterpolator(interpolation_method="linear")
        
        all_frames = interpolator.interpolate_between_keyframes(
            keyframes=keyframes,
            num_interpolations=args.interpolations,
            save_dir=str(frames_dir)
        )
    else:
        all_frames = keyframes
    # Step 4: Create signal mapping
    print("\n" + "="*70)
    print("STEP 4: Creating signal mapping")
    print("="*70)
    
    mapper = SignalMapper(num_frames=len(all_frames))
    
    # Save mapping
    mapping_file = output_dir / "signal_mapping.json"
    mapper.save_mapping(str(mapping_file))
    
    print(f"Frames mapped to signals 0.0 to 1.0")
    print(f"Mapping resolution: {1.0 / (len(all_frames) - 1):.6f} per frame")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'initial_image': args.initial_image,
        'path_prompts': args.path_prompts,
        'num_prompts': len(evolutionary_prompts),
        'num_keyframes': len(keyframes),
        'num_interpolations': args.interpolations,
        'total_frames': len(all_frames),
        'parameters': {
            'strength': args.strength,
            'guidance_scale': args.guidance,
            'num_inference_steps': args.steps,
            'width': args.width,
            'height': args.height,
            'seed': args.seed
        },
        'lora_used': args.lora is not None,
        'lora_path': args.lora,
        'rag_used': args.use_rag,
        'txt2img_used': args.use_txt2img
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
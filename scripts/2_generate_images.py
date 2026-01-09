"""
FILE: scripts/2_generate_images.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from PIL import Image
from src.vlm.prompt_generator import PromptGenerator
from src.sdxl.generator import SDXLGenerator
from src.sdxl.latent_slerp_interpolator import LatentSlerpInterpolator
from src.sdxl.latent_interpolator import LatentInterpolator
from src.sdxl.mapper import SignalMapper
from config.model_config import ModelConfig
from config.paths import Paths
import json
import os
import gc
import torch
from datetime import datetime
import time
from collections import defaultdict
import numpy as np



def main():
    parser = argparse.ArgumentParser(
        description="Generate images using VLM + SDXL"
    )
    
    # Input options
    parser.add_argument('--initial-image', type=str, required=True,
                       help='Path to initial/target image')
    parser.add_argument('--evolution', type=str, required=True,
                       help='Evolution description (e.g., "dot becomes a neuron")')
    parser.add_argument('--prompts-type', type=str, default='basic', 
                        choices=['basic', 'few_shot', 'matrix', 'only'],
                        help='Type of the evolution prompt.')
    
    # Generation parameters
    parser.add_argument('--num-prompts', type=int, default=13,
                       help='Number of evolutionary prompts (max sequence length)')
    parser.add_argument('--num-keyframes', type=int, default=None,
                       help='Number of keyframes to generate (default: same as prompts)')
    parser.add_argument('--interpolate', action='store_true',
                        help='Whether to interpolate between keyframes')
    parser.add_argument('--interpolations', type=int, default=None,
                       help='Interpolations between keyframes')
    parser.add_argument('--interpolator', type=str, default='linear',
                       choices=['linear', 'cosine', 'latent_slerp'],
                       help='Interpolation method (linear, cosine, latent_slerp)')
    parser.add_argument('--analysis-type', type=str, default='detailed',
                        help='Type of the analysis for the initial image (detailed, brief, technical)')
    
    # SDXL parameters
    parser.add_argument('--strength', type=float, default=0.45,
                       help='SDXL strength (0.0-1.0)')
    parser.add_argument('--guidance', type=float, default=18.0,
                       help='Guidance scale')
    parser.add_argument('--steps', type=int, default=50,
                       help='Inference steps')
    parser.add_argument('--width', type=int, default=512,
                       help='Image width')
    parser.add_argument('--height', type=int, default=512,
                       help='Image height')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    # Options
    parser.add_argument('--use-txt2img', action='store_true',
                       help='Start with txt2img instead of initial image')
    parser.add_argument('--txt2img-prompt', type=str, default=None,
                       help='Prompt for txt2img (if --use-txt2img)')
    parser.add_argument('--lora', type=str, default=None,
                       help='Path to LoRA weights')
    parser.add_argument('--use-rag', action='store_true',
                       help='Use RAG for guideline enhancement')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory name')
    parser.add_argument('--save-prompts-only', action='store_true',
                       help='Only generate and save prompts, no image generation')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Image Generation Pipeline")
    print("="*70)
    
    timings = defaultdict(list)
    t_global_start = time.perf_counter()


    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.splitext(os.path.basename(args.initial_image))[0]
    output_name = args.output or f"generation_{filename}_{timestamp}"
    output_dir = Paths.OUTPUT_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Step 1: Generate evolutionary prompts with VLM
    print("\n" + "="*70)
    print("STEP 1: Generating evolutionary prompts with VLM")
    print("="*70)
    
    t_vlm_start = time.perf_counter()
    vlm = PromptGenerator(
        model_id=ModelConfig.VLM_MODEL_ID,
        device=ModelConfig.VLM_DEVICE
    )
    
    # Get visual guidelines if using RAG
    visual_guidelines = None
    if args.use_rag:
        try:
            from src.rag.guideline_rag import VisualGuidelineRAG
            rag = VisualGuidelineRAG()
            guidelines = rag.retrieve(args.evolution, top_k=3)
            if guidelines:
                visual_guidelines = "\n".join([g.text for g in guidelines])
                print(f"\nRetrieved {len(guidelines)} guidelines from RAG")
        except Exception as e:
            print(f"Warning: Could not load RAG: {e}")
    
    style_guidelines={
        "background": "white background",
        "color_palette": "white, pink, purple",
        "style": "minimalism, line art, vector art, modern graphic design, abstract art, organic form",
    }
    # Generate prompts
    evolutionary_prompts = vlm.generate_evolutionary_sequence(
        image_path=args.initial_image,
        evolution_description=args.evolution,
        num_prompts=args.num_prompts,
        visual_guidelines=visual_guidelines,
        analysis_type=args.analysis_type,
        prompts_type=args.prompts_type, 
        style_guidelines=style_guidelines
    )
    
    # Save prompts
    prompts_file = output_dir / "prompts.json"
    vlm.save_prompts(evolutionary_prompts, str(prompts_file))
    vlm.display_prompts(evolutionary_prompts)
    
    t_vlm_end = time.perf_counter()
    timings["vlm_time"].append(t_vlm_end - t_vlm_start)
    
    # Exit if only generating prompts
    if args.save_prompts_only:
        print(f"\n✓ Prompts saved to: {prompts_file}")
        print("Exiting (--save-prompts-only specified)")
        return
    
    # free memory 
    # del analyzer
    # torch.cuda.empty_cache()
    # gc.collect()
    
    
    
    # Step 2: Generate keyframes with SDXL
    print("\n" + "="*70)
    print("STEP 2: Generating keyframes with SDXL")
    print("="*70)
    
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
        txt2img_prompt = args.txt2img_prompt or selected_prompts[0]
        initial_image = sdxl.generate_initial_image(
            prompt=txt2img_prompt,
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
    
    # double each prompt to slow evolution
    # selected_prompts = [p for p in selected_prompts for _ in (0, 1)]
    
    # Generate keyframes and time the operation
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    
    keyframes = sdxl.generate_keyframes_from_prompts(
        initial_image=initial_image,
        prompts=selected_prompts,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        seed=args.seed,
        save_dir=str(frames_dir)
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_keyframe_time = t1 - t0
    mean_keyframe_time = total_keyframe_time / len(keyframes)

    timings["keyframe_total_time"].append(total_keyframe_time)
    timings["keyframe_mean_time"].append(mean_keyframe_time)

    if args.interpolate:

        # Step 3: Interpolate frames
        print("\n" + "="*70)
        print("STEP 3: Interpolating frames")
        print("="*70)

        # Create appropriate interpolator based on user choice
        if args.interpolator == "latent_slerp":
            interpolator = LatentSlerpInterpolator(
                vae=sdxl.img2img_pipe.vae,
                device=ModelConfig.SDXL_DEVICE
            )
        else:
            # latent=True makes linear/cosine comparable to SLERP
            interpolator = LatentInterpolator(
                interpolation_method=args.interpolator,  # "linear" or "cosine"
                latent=True,
                vae=sdxl.img2img_pipe.vae,
                device=ModelConfig.SDXL_DEVICE,
            )

        # time interpolation
        t0 = time.perf_counter()

        # all_frames = interpolator.interpolate_between_keyframes(
        #     keyframes=keyframes,
        #     num_interpolations=args.interpolations,
        #     save_dir=str(frames_dir)
        # )
        all_frames = interpolator.interpolate_between_keyframes(
            keyframes=keyframes,
            num_interpolations=args.interpolations,
            save_dir=str(frames_dir)
        )
        interp_norm_stats = interpolator.get_norm_stats()
        interp_decode_stats = interpolator.get_decode_stats()
        interp_encode_stats = interpolator.get_encode_stats()
        interpolator.unload()

        t1 = time.perf_counter()
        
    

        num_interp_frames = len(all_frames) - len(keyframes)
        interp_time_total = t1 - t0
        interp_time_per_frame = interp_time_total / max(1, num_interp_frames)

        timings["interp_total_time"].append(interp_time_total)
        timings["interp_time_per_frame"].append(interp_time_per_frame)

        # if args.interpolator == 'latent_slerp':
        #     interpolator.unload()
        #     decode_stats = interpolator.get_decode_stats()
        #     timings["vae_decode_mean"].append(decode_stats["mean_decode_time"])
            
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
    
    # save timings 
    t_global_end = time.perf_counter()
    timings["full_offline_time"].append(t_global_end - t_global_start)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'initial_image': args.initial_image,
        'evolution': args.evolution,
        'num_prompts': len(evolutionary_prompts),
        'num_keyframes': len(keyframes),
        'num_interpolations': args.interpolations,
        'interpolator': args.interpolator if args.interpolate else None,
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
    metadata["interpolation_stats"] = {
        "norm": interp_norm_stats,
        "decode": interp_decode_stats,
        "encode": interp_encode_stats,
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    timing_file = output_dir / "timings.json"
    with open(timing_file, "w") as f:
        json.dump({k: float(np.mean(v)) for k, v in timings.items()}, f, indent=2)

    print(f"Timing metrics saved to: {timing_file}")

    

    print("\n" + "="*70)
    print("✓ IMAGE GENERATION COMPLETE")
    print("="*70)
    print(f"\nResults:")
    print(f"  Output directory: {output_dir}")
    print(f"  Keyframes: {len(keyframes)}")
    print(f"  Total frames: {len(all_frames)}")
    print(f"  Prompts: {prompts_file}")
    print(f"  Mapping: {mapping_file}")
    print(f"  Metadata: {metadata_file}")
    print("\n" + "="*70)
    print("\nNext step: Create video with signal")
    print(f"  python scripts/3_create_video.py --project {output_name} --signal sine")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
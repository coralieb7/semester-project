"""
FILE: scripts/2_generate_universal.py
Generate images using Universal Evolution System + SDXL

This script uses the UniversalVisualEvolution class which works for ANY visual type:
- Neurons, particles, networks
- Abstract art, paintings
- Minimalist graphics, logos
- Geometric patterns, vectors
- Literally anything!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from PIL import Image
from src.vlm.universal_evolution_generator import UniversalEvolutionGenerator
from src.sdxl.generator import SDXLGenerator
from src.sdxl.interpolator import FrameInterpolator
from src.sdxl.mapper import SignalMapper
from config.model_config import ModelConfig
from config.paths import Paths
import json
import os, gc, torch
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Universal Evolution + SDXL"
    )

    # Input options
    parser.add_argument('--initial-image', type=str, required=True,
                       help='Path to initial/target image')
    parser.add_argument('--evolution-template', type=str,
                       default='grow_and_complexify',
                       help='Evolution template (grow_and_complexify, simplify_and_shrink, densify, expand, intensify, fade)')

    # Generation parameters
    parser.add_argument('--num-prompts', type=int, default=13,
                       help='Number of evolutionary prompts (max sequence length)')
    parser.add_argument('--num-keyframes', type=int, default=None,
                       help='Number of keyframes to generate (default: same as prompts)')
    parser.add_argument('--interpolate', action='store_true',
                        help='Whether to interpolate between keyframes')
    parser.add_argument('--interpolations', type=int, default=None,
                       help='Interpolations between keyframes')
    parser.add_argument('--style-prefix', type=str, default="",
                        help='Style prefix for LoRA activation (e.g., "TRGR, style")')

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
    parser.add_argument('--txt2img-prompt', type=str, default=None,
                       help='Prompt for txt2img (if --use-txt2img)')
    parser.add_argument('--lora', type=str, default=None,
                       help='Path to LoRA weights')

    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory name')
    parser.add_argument('--save-prompts-only', action='store_true',
                       help='Only generate and save prompts, no image generation')
    parser.add_argument('--list-templates', action='store_true',
                       help='List available evolution templates and exit')

    args = parser.parse_args()

    # List templates if requested
    if args.list_templates:
        UniversalEvolutionGenerator.list_templates()
        return

    print("="*70)
    print("Universal Evolution Image Generation Pipeline")
    print("="*70)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.splitext(os.path.basename(args.initial_image))[0]
    output_name = args.output or f"generation_universal_{filename}_{timestamp}"
    output_dir = Paths.OUTPUT_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Step 1: Generate evolutionary prompts with Universal Evolution
    print("\n" + "="*70)
    print("STEP 1: Generating universal evolution sequence")
    print("="*70)

    # Initialize VLM
    # vlm = ImageAnalyzer(
    #     model_id=ModelConfig.VLM_MODEL_ID,
    #     device=ModelConfig.VLM_DEVICE
    # )

    # Initialize Universal Evolution
    vlm = UniversalEvolutionGenerator(
        model_id=ModelConfig.VLM_MODEL_ID,
        device=ModelConfig.VLM_DEVICE
    )

    # Generate evolution sequence
    evolutionary_prompts = vlm.generate_evolution_sequence(
        image_path=args.initial_image,
        evolution_template=args.evolution_template,
        num_steps=args.num_prompts,
        style_prefix=args.style_prefix
    )

    vlm.save_prompts(evolutionary_prompts,
                           str(output_dir / "prompts.json"))

    print(f"\n✓ Prompts saved to: prompts.json")

    # Display prompts
    print("\nGenerated Prompts:")
    print("-" * 70)
    for p in evolutionary_prompts:
        print(f"\nStep {p['step']}: (signal={p['signal']:.2f})")
        print(f"  {p['prompt'][:150]}..." if len(p['prompt']) > 150 else f"  {p['prompt']}")
        if p['change']:
            print(f"  Change: {p['change']}")
    print("-" * 70)

    # Exit if only generating prompts
    if args.save_prompts_only:
        print(f"\n✓ Prompts saved to: prompts.json")
        print("Exiting (--save-prompts-only specified)")
        return
    
    del vlm
    gc.collect()
    torch.cuda.empty_cache()

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
        'evolution_template': args.evolution_template,
        'style_prefix': args.style_prefix,
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
        'txt2img_used': args.use_txt2img,
        'universal_evolution': True
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*70)
    print("✓ UNIVERSAL EVOLUTION GENERATION COMPLETE")
    print("="*70)
    print(f"\nResults:")
    print(f"  Output directory: {output_dir}")
    print(f"  Evolution template: {args.evolution_template}")
    print(f"  Keyframes: {len(keyframes)}")
    print(f"  Total frames: {len(all_frames)}")
    print(f"  Mapping: {mapping_file}")
    print(f"  Metadata: {metadata_file}")
    print("\n" + "="*70)
    print("\nNext step: Create video with signal")
    print(f"  python scripts/3_create_video.py --project {output_name} --signal sine")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

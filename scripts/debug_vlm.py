"""
FILE: scripts/debug_vlm.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.vlm.image_analyser import ImageAnalyzer
from src.vlm.prompt_generator import PromptGenerator
from config.model_config import ModelConfig
import json


def main():
    parser = argparse.ArgumentParser(
        description="Debug and test VLM functionality"
    )
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to test image')
    parser.add_argument('--mode', type=str, default='analyze',
                       choices=['analyze', 'prompts', 'compare', 'features'],
                       help='Debug mode')
    parser.add_argument('--image2', type=str, default=None,
                       help='Second image for comparison')
    parser.add_argument('--evolution', type=str, default=None,
                       help='Evolution description for prompt generation')
    parser.add_argument('--num-prompts', type=int, default=5,
                       help='Number of prompts to generate')
    parser.add_argument('--analysis-type', type=str, default='brief',
                       choices=['detailed', 'brief', 'technical'],
                       help='Type of analysis')
    parser.add_argument('--path-save', type=str, default=None,
                       help='Save output to file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("VLM Debug Tool")
    print("="*70)
    
    if args.mode == 'analyze':
        print(f"\nMode: Image Analysis ({args.analysis_type})")
        print(f"Image: {args.image}")
        
        analyzer = ImageAnalyzer(
            model_id=ModelConfig.VLM_MODEL_ID,
            device=ModelConfig.VLM_DEVICE
        )
        
        result = analyzer.analyze_image(args.image, args.analysis_type)
        
        print("\n" + "="*70)
        print("ANALYSIS RESULT")
        print("="*70)
        print(result['description'])
        print("="*70)
        
        if args.save:
            with open(args.save, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Saved to: {args.save}")
    
    elif args.mode == 'features':
        print(f"\nMode: Feature Extraction")
        print(f"Image: {args.image}")
        
        analyzer = ImageAnalyzer(
            model_id=ModelConfig.VLM_MODEL_ID,
            device=ModelConfig.VLM_DEVICE
        )
        
        features = analyzer.extract_key_features(args.image)
        
        print("\n" + "="*70)
        print("EXTRACTED FEATURES")
        print("="*70)
        for key, value in features.items():
            print(f"{key.capitalize()}: {value}")
        print("="*70)
        
        if args.save:
            with open(args.save, 'w', encoding='utf-8') as f:
                json.dump(features, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Saved to: {args.save}")
    
    elif args.mode == 'compare':
        if not args.image2:
            print("ERROR: --image2 required for comparison mode")
            sys.exit(1)
        
        print(f"\nMode: Image Comparison")
        print(f"Image 1: {args.image}")
        print(f"Image 2: {args.image2}")
        
        analyzer = ImageAnalyzer(
            model_id=ModelConfig.VLM_MODEL_ID,
            device=ModelConfig.VLM_DEVICE
        )
        
        comparison = analyzer.compare_images(args.image, args.image2)
        
        print("\n" + "="*70)
        print("COMPARISON RESULT")
        print("="*70)
        print(comparison)
        print("="*70)
        
        if args.save:
            with open(args.save, 'w', encoding='utf-8') as f:
                f.write(comparison)
            print(f"\n✓ Saved to: {args.save}")
    
    elif args.mode == 'prompts':
        if not args.evolution:
            print("ERROR: --evolution required for prompt generation mode")
            sys.exit(1)
        
        print(f"\nMode: Prompt Generation")
        print(f"Image: {args.image}")
        print(f"Evolution: {args.evolution}")
        print(f"Number of prompts: {args.num_prompts}")
        
        generator = PromptGenerator(
            model_id=ModelConfig.VLM_MODEL_ID,
            device=ModelConfig.VLM_DEVICE
        )
        
        prompts = generator.generate_evolutionary_sequence(
            image_path=args.image,
            evolution_description=args.evolution,
            num_prompts=args.num_prompts,
            analysis_type=args.analysis_type
        )
        
        generator.display_prompts(prompts)
        
        if args.path_save:
            generator.save_prompts(prompts, args.path_save)
    
    print("\n✓ Debug complete")


if __name__ == "__main__":
    main()
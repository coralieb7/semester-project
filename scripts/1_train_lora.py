"""
FILE: scripts/1_train_lora.py - FIXED
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.sdxl.lora_trainer import LoRATrainer, ImageCaptionDataset
from config.paths import Paths
from config.model_config import ModelConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train LoRA for SDXL with custom dataset"
    )
    
    # Dataset parameters
    parser.add_argument('--dataset-dir', type=str,
                       default=str(Paths.TRAINING_DATA_DIR),
                       help='Directory containing images and pairing.json')
    parser.add_argument('--pairing-file', type=str, default='pairing.json',
                       help='Name of pairing JSON file')
    
    # Training parameters
    parser.add_argument('--output-dir', type=str,
                       default=str(Paths.LORA_WEIGHTS_DIR / 'custom_lora'),
                       help='Output directory for LoRA weights')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (keep at 1 for 1024x1024)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # LoRA parameters
    parser.add_argument('--lora-rank', type=int, default=16,
                       help='LoRA rank (8-32 typical)')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='LoRA alpha (usually 2x rank)')
    
    # Image parameters
    parser.add_argument('--image-size', type=int, default=1024,
                       help='Training image size')
    parser.add_argument('--no-center-crop', action='store_true',
                       help='Use random crop instead of center crop')
    
    # Optimization
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='Max gradient norm for clipping')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LoRA Training for SDXL")
    print("="*70)
    
    # Load dataset
    print(f"\nLoading dataset from: {args.dataset_dir}")
    
    try:
        dataset = ImageCaptionDataset(
            dataset_dir=args.dataset_dir,
            pairing_file=args.pairing_file,
            size=args.image_size,
            center_crop=not args.no_center_crop
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"\nMake sure you have:")
        print(f"  1. Images in: {args.dataset_dir}")
        print(f"  2. Pairing file: {args.dataset_dir}/{args.pairing_file}")
        sys.exit(1)
    
    if len(dataset) < 5:
        print(f"\nWarning: Only {len(dataset)} training samples found.")
        print("LoRA typically needs at least 10-20 images for good results.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Initialize trainer
    print("\n" + "="*70)
    print("Initializing LoRA Trainer")
    print("="*70)
    
    trainer = LoRATrainer(
        model_id=ModelConfig.SDXL_MODEL_ID,
        device=ModelConfig.SDXL_DEVICE,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )
    
    # Train - FIXED: removed use_8bit_adam and mixed_precision
    trainer.train(
        dataset=dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation,
        save_every=args.save_every,
        max_grad_norm=args.max_grad_norm
    )
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"\nLoRA weights saved to: {args.output_dir}")
    print(f"\nTo use in generation:")
    print(f"  python scripts/2_generate_images.py \\")
    print(f"    --image data/input/your_image.png \\")
    print(f"    --evolution 'your evolution' \\")
    print(f"    --lora {args.output_dir}/checkpoint_epoch_{args.epochs}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
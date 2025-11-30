"""
FILE: scripts/debug_training_data.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.sdxl.lora_trainer import ImageCaptionDataset
import matplotlib.pyplot as plt
from config.paths import Paths

def main():
    print("="*70)
    print("Debug Training Data")
    print("="*70)
    
    # Load dataset
    dataset = ImageCaptionDataset(
        dataset_dir=str(Paths.TRAINING_DATA_DIR),
        pairing_file="pairing.json",
        size=1024,
        center_crop=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Check first few samples
    print("\nChecking first 5 samples:")
    for i in range(min(10, len(dataset))):
        try:
            sample = dataset[i]
            image = sample['image']
            caption = sample['caption']
            
            print(f"\n[{i}] Caption: {caption[:200]}")
            print(f"    Image shape: {image.shape}")
            print(f"    Image dtype: {image.dtype}")
            print(f"    Image range: [{image.min():.3f}, {image.max():.3f}]")
            
            # Check for NaN or Inf
            if torch.isnan(image).any():
                print("    ⚠️  WARNING: Image contains NaN!")
            if torch.isinf(image).any():
                print("    ⚠️  WARNING: Image contains Inf!")
                
        except Exception as e:
            print(f"\n[{i}] ERROR: {e}")
    
    # # Visualize a few samples
    # print("\nVisualizing samples...")
    # fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # axes = axes.flatten()
    
    # for i in range(min(6, len(dataset))):
    #     sample = dataset[i]
    #     image = sample['image']
    #     caption = sample['caption']
        
    #     # Denormalize image for display
    #     image_display = (image.permute(1, 2, 0).numpy() + 1) / 2
    #     image_display = image_display.clip(0, 1)
        
    #     axes[i].imshow(image_display)
    #     axes[i].set_title(caption[:40] + "...", fontsize=8)
    #     axes[i].axis('off')
    
    # plt.tight_layout()
    # output_path = Paths.OUTPUT_DIR / "training_data_samples.png"
    # plt.savefig(output_path, dpi=150, bbox_inches='tight')
    # print(f"\n✓ Sample visualization saved to: {output_path}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
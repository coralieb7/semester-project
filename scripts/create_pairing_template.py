
"""
FILE: scripts/create_pairing_template.py
HELPER SCRIPT: Generate template pairing.json from existing images
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from config.paths import Paths


def create_pairing_template(image_dir: str, output_file: str = "pairing.json"):
    """Create a template pairing.json from images in directory"""
    
    image_dir = Path(image_dir)
    
    # Find all images
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
    images = []
    for ext in image_extensions:
        images.extend(image_dir.glob(f'*{ext}'))
        images.extend(image_dir.glob(f'*{ext.upper()}'))
    
    if not images:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(images)} images")
    
    # Create pairing dictionary
    pairing = {}
    
    for i, img_path in enumerate(sorted(images), 1):
        key = f"image{i}"
        
        # Template prompt - user should fill this in
        template_prompt = (
            f"minimalist graphic design of [DESCRIBE SUBJECT], "
            f"centered composition on [DESCRIBE BACKGROUND], "
            f"aesthetic balance and clarity"
        )
        
        pairing[key] = {
            "image_path": str(img_path.absolute()),
            "text_prompt": template_prompt
        }
    
    # Save to JSON
    output_path = image_dir / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairing, f, indent=4, ensure_ascii=False)
    
    print(f"\n✓ Template pairing file created: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Open {output_path}")
    print(f"2. Replace the template text_prompts with actual descriptions")
    print(f"3. Run training: python scripts/1_train_lora.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create pairing.json template")
    parser.add_argument('--image-dir', type=str,
                       default=str(Paths.TRAINING_DATA_DIR),
                       help='Directory containing images')
    parser.add_argument('--output', type=str, default='pairing.json',
                       help='Output filename')
    
    args = parser.parse_args()
    create_pairing_template(args.image_dir, args.output)
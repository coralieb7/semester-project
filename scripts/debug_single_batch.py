"""
FILE: scripts/debug_single_batch.py - FIXED
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.sdxl.lora_trainer import ImageCaptionDataset, LoRATrainer
from config.paths import Paths

# Load one batch
dataset = ImageCaptionDataset(
    dataset_dir=str(Paths.TRAINING_DATA_DIR),
    size=768
)

print(f"Dataset size: {len(dataset)}")

# Get first sample
sample = dataset[0]
print(f"\nFirst sample:")
print(f"  Image shape: {sample['image'].shape}")
print(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
print(f"  Caption: {sample['caption'][:80]}")

# Check for NaN/Inf in image
if torch.isnan(sample['image']).any():
    print("  ⚠️  IMAGE CONTAINS NaN!")
if torch.isinf(sample['image']).any():
    print("  ⚠️  IMAGE CONTAINS Inf!")

# Initialize trainer
trainer = LoRATrainer(lora_rank=8)

# Test encoding one prompt
print("\nTesting prompt encoding...")
test_caption = [sample['caption']]
prompt_embeds, pooled_embeds = trainer.encode_prompt(test_caption)

print(f"Prompt embeds shape: {prompt_embeds.shape}")
print(f"Prompt embeds range: [{prompt_embeds.min():.3f}, {prompt_embeds.max():.3f}]")
print(f"Pooled embeds shape: {pooled_embeds.shape}")

if torch.isnan(prompt_embeds).any():
    print("⚠️  PROMPT EMBEDS CONTAIN NaN!")

# Test VAE encoding - USE FLOAT32 FOR VAE
print("\nTesting VAE encoding...")
test_image = sample['image'].unsqueeze(0).to('cuda', dtype=torch.float32)  # Changed to float32

with torch.no_grad():
    latents = trainer.vae.encode(test_image).latent_dist.sample()
    latents = latents * 0.13025

print(f"Latents shape: {latents.shape}")
print(f"Latents range: [{latents.min():.3f}, {latents.max():.3f}]")

if torch.isnan(latents).any():
    print("⚠️  LATENTS CONTAIN NaN!")
else:
    print("✓ Latents are clean (no NaN)")

print("\n✓ Debug complete - ready to train!")
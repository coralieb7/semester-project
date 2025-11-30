"""
FILE: src/sdxl/lora_trainer.py - VAE FIX
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionXLPipeline, DDPMScheduler, AutoencoderKL
from peft import LoraConfig, get_peft_model
from PIL import Image
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import json
import os
import gc


class ImageCaptionDataset(Dataset):
    """Dataset for LoRA training with JSON pairing file"""
    
    def __init__(self, 
                 dataset_dir: str, 
                 pairing_file: str = "pairing.json",
                 size: int = 1024, 
                 center_crop: bool = True):
        self.dataset_dir = Path(dataset_dir)
        self.size = size
        
        pairing_path = self.dataset_dir / pairing_file
        if not pairing_path.exists():
            raise FileNotFoundError(f"Pairing file not found: {pairing_path}")
        
        print(f"Loading pairing file: {pairing_path}")
        with open(pairing_path, 'r', encoding='utf-8') as f:
            self.pairings = json.load(f)
        
        self.image_paths = []
        self.captions = []
        
        for key, data in self.pairings.items():
            image_path = Path(data['image_path'])
            
            if not image_path.exists():
                image_path = self.dataset_dir / image_path.name
            
            if image_path.exists():
                self.image_paths.append(image_path)
                self.captions.append(data['text_prompt'])
            else:
                print(f"Warning: Image not found: {image_path}")
        
        if len(self.image_paths) == 0:
            raise ValueError("No valid image-caption pairs found!")
        
        print(f"✓ Loaded {len(self.image_paths)} image-caption pairs")
        
        # Modified transform - no normalization, VAE expects [0, 1]
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),  # Converts to [0, 1]
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = self.transform(image)
            
            # Normalize to [-1, 1] for VAE
            image = 2.0 * image - 1.0
            
            caption = self.captions[idx]
            return {'image': image, 'caption': caption}
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))


class LoRATrainer:
    """Train LoRA for SDXL - VAE FIXED"""
    
    def __init__(self, 
                 model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 device: str = 'cuda',
                 lora_rank: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1):
        
        print(f"Initializing LoRA Trainer...")
        print(f"  Model: {model_id}")
        print(f"  LoRA rank: {lora_rank}")
        
        self.device = device
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Load SDXL pipeline
        print("Loading SDXL model...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        
        # Extract components
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder.to(device)
        self.text_encoder_2 = pipe.text_encoder_2.to(device)
        self.unet = pipe.unet.to(device)
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        
        # **CRITICAL FIX**: Keep VAE in float32 to avoid NaN
        self.vae = self.vae.to(device, dtype=torch.float32)
        
        # Training scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_id, 
            subfolder="scheduler"
        )
        
        # Disable VAE tiling/slicing - can cause NaN
        # self.vae.enable_tiling()
        # self.vae.enable_slicing()
        
        # Freeze everything except UNet
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # Add LoRA to UNet
        print("Adding LoRA layers...")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=lora_dropout,
        )
        
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        print("✓ LoRA Trainer initialized")
        print("✓ VAE in float32 to prevent NaN")
    
    def encode_prompt(self, prompts):
        """Encode prompts"""
        
        # Tokenize
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        text_inputs_2 = self.tokenizer_2(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode
        with torch.no_grad():
            text_input_ids = text_inputs.input_ids.to(self.device)
            encoder_hidden_states = self.text_encoder(
                text_input_ids,
                output_hidden_states=True,
            )
            prompt_embeds_1 = encoder_hidden_states.hidden_states[-2]
            
            text_input_ids_2 = text_inputs_2.input_ids.to(self.device)
            encoder_hidden_states_2 = self.text_encoder_2(
                text_input_ids_2,
                output_hidden_states=True,
            )
            pooled_prompt_embeds = encoder_hidden_states_2[0]
            prompt_embeds_2 = encoder_hidden_states_2.hidden_states[-2]
            
            prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        
        return prompt_embeds, pooled_prompt_embeds
    
    def train(self, 
              dataset: Dataset, 
              output_dir: str, 
              num_epochs: int = 10, 
              batch_size: int = 1, 
              learning_rate: float = 1e-4,
              gradient_accumulation_steps: int = 4,
              save_every: int = 5,
              max_grad_norm: float = 1.0):
        """Train LoRA - VAE in FP32"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Set modes
        self.unet.train()
        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        
        print("\n" + "="*70)
        print("Starting LoRA Training")
        print("="*70)
        print(f"Dataset: {len(dataset)} samples")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation: {gradient_accumulation_steps}")
        print(f"Learning rate: {learning_rate}")
        print("="*70 + "\n")
        
        global_step = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_valid = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                try:
                    # Get batch - convert to float32 for VAE
                    images = batch['image'].to(self.device, dtype=torch.float32)
                    captions = batch['caption']
                    bsz = images.shape[0]
                    
                    # Encode text
                    prompt_embeds, pooled_prompt_embeds = self.encode_prompt(captions)
                    
                    # Encode images with VAE (fp32)
                    with torch.no_grad():
                        latents = self.vae.encode(images).latent_dist.sample()
                        latents = latents * 0.13025
                        # Convert to fp16 for UNet
                        latents = latents.to(dtype=torch.float16)
                    
                    # Check for NaN after VAE
                    if torch.isnan(latents).any():
                        print(f"\n⚠️  VAE produced NaN at step {step}, skipping")
                        continue
                    
                    # Sample noise
                    noise = torch.randn_like(latents)
                    
                    # Random timestep
                    timesteps = torch.randint(
                        0, 
                        self.noise_scheduler.config.num_train_timesteps,
                        (bsz,), 
                        device=self.device
                    ).long()
                    
                    # Add noise
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # SDXL conditioning
                    add_time_ids = torch.tensor(
                        [[images.shape[-1], images.shape[-1], 0, 0, images.shape[-1], images.shape[-1]]]
                    ).repeat(bsz, 1).to(self.device, dtype=prompt_embeds.dtype)
                    
                    added_cond_kwargs = {
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids
                    }
                    
                    # Predict noise
                    model_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False
                    )[0]
                    
                    # Loss
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    # Check for NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\n⚠️  NaN/Inf loss at step {step}")
                        torch.cuda.empty_cache()
                        continue
                    
                    # Scale loss
                    loss = loss / gradient_accumulation_steps
                    
                    # Backward
                    loss.backward()
                    
                    # Update weights
                    if (step + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.unet.parameters(), 
                            max_grad_norm
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                    
                    # Track
                    loss_val = loss.item() * gradient_accumulation_steps
                    epoch_loss += loss_val
                    num_valid += 1
                    
                    # Update bar
                    if num_valid > 0:
                        progress_bar.set_postfix({
                            'loss': f'{loss_val:.4f}',
                            'avg': f'{epoch_loss/num_valid:.4f}'
                        })
                    
                    # Periodic cleanup
                    if step % 20 == 0:
                        torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n⚠️  OOM at step {step}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e
            
            # Epoch summary
            if num_valid > 0:
                avg_loss = epoch_loss / num_valid
                print(f"\n✓ Epoch {epoch+1}: avg loss = {avg_loss:.4f} ({num_valid}/{len(dataloader)} steps)")
            else:
                print(f"\n✗ Epoch {epoch+1}: No valid steps!")
                break
            
            # Save
            if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
                if num_valid > 0:
                    ckpt = Path(output_dir) / f"checkpoint_epoch_{epoch+1}"
                    self.save_lora(str(ckpt))
            
            torch.cuda.empty_cache()
            gc.collect()
        
        print("\n" + "="*70)
        print("✓ Training complete!")
        print("="*70)
    
    def save_lora(self, output_dir: str):
        """Save LoRA weights"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.unet.save_pretrained(output_dir)
        
        config = {
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
        }
        with open(Path(output_dir) / "lora_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Saved: {output_dir}")
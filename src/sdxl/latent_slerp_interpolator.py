"""
FILE: src/sdxl/latent_slerp_interpolator.py - FIXED (Black Image Issue)

Latent space SLERP interpolator for SDXL
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Optional
from tqdm import tqdm
from pathlib import Path
from diffusers import AutoencoderKL


class LatentSlerpInterpolator:
    """Interpolate frames in SDXL latent space using SLERP"""

    def __init__(self, vae: AutoencoderKL, device: str = "cuda"):
        """
        Initialize latent SLERP interpolator

        Args:
            vae: SDXL VAE encoder/decoder
            device: Device to run on
        """
        self.vae = vae
        self.device = device
        
        # CRITICAL: Keep VAE in float32 for stable encode/decode
        self.vae.to(device=device, dtype=torch.float32)
        self.vae.eval()
        
        print(f"✓ Latent SLERP Interpolator initialized (VAE in float32)")

    def slerp(self, v0: torch.Tensor, v1: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
        """
        Spherical linear interpolation between two tensors

        Args:
            v0: Starting tensor
            v1: Ending tensor
            t: Interpolation parameter [0, 1]
            eps: Small value to prevent division by zero

        Returns:
            Interpolated tensor
        """
        # Store original shape
        original_shape = v0.shape
        
        # Flatten to vectors for dot product
        v0_flat = v0.flatten()
        v1_flat = v1.flatten()

        # Normalize
        v0_norm = v0_flat / (torch.norm(v0_flat) + eps)
        v1_norm = v1_flat / (torch.norm(v1_flat) + eps)

        # Calculate angle between vectors
        dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
        theta = torch.acos(dot)

        # If angle is very small, use linear interpolation
        if theta.abs() < eps:
            result = (1.0 - t) * v0_flat + t * v1_flat
        else:
            # SLERP formula
            sin_theta = torch.sin(theta)
            w0 = torch.sin((1.0 - t) * theta) / sin_theta
            w1 = torch.sin(t * theta) / sin_theta
            result = w0 * v0_flat + w1 * v1_flat

        return result.reshape(original_shape)

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode PIL image to latent space

        Args:
            image: PIL Image

        Returns:
            Latent tensor
        """
        # Convert to tensor [0, 1]
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        # Move to device in float32
        img_tensor = img_tensor.to(device=self.device, dtype=torch.float32)

        # Normalize to [-1, 1]
        img_tensor = 2.0 * img_tensor - 1.0

        # Encode
        with torch.no_grad():
            latent_dist = self.vae.encode(img_tensor).latent_dist
            latent = latent_dist.sample()
            
            # Apply SDXL scaling
            latent = latent * 0.13025

        return latent

    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """
        Decode latent to PIL image

        Args:
            latent: Latent tensor

        Returns:
            PIL Image
        """
        with torch.no_grad():
            # Unscale
            latent = latent / 0.13025
            
            # Decode
            decoded = self.vae.decode(latent).sample
            
            # Denormalize from [-1, 1] to [0, 1]
            decoded = (decoded + 1.0) / 2.0
            decoded = torch.clamp(decoded, 0.0, 1.0)

        # Convert to numpy and PIL
        decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
        decoded = (decoded * 255).astype(np.uint8)

        return Image.fromarray(decoded)

    def interpolate_between_keyframes(self,
                                     keyframes: List[Image.Image],
                                     num_interpolations: int = 5,
                                     save_dir: Optional[str] = None,
                                     debug: bool = False) -> List[Image.Image]:
        """
        Interpolate frames between all keyframes using latent SLERP

        Args:
            keyframes: List of keyframe images
            num_interpolations: Number of frames to add between each pair
            save_dir: Optional directory to save interpolated frames
            debug: Print debug information

        Returns:
            List of all frames (keyframes + interpolated)
        """
        print(f"\n{'='*70}")
        print(f"Interpolating frames with Latent SLERP")
        print(f"{'='*70}")
        print(f"Keyframes: {len(keyframes)}")
        print(f"Interpolations per pair: {num_interpolations}")
        print(f"VAE dtype: {next(self.vae.parameters()).dtype}")

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        all_frames = []
        frame_idx = 0

        # Process each pair of keyframes
        for i in tqdm(range(len(keyframes) - 1), desc="Interpolating keyframe pairs"):
            
            if debug:
                print(f"\n[DEBUG] Processing pair {i} -> {i+1}")
            
            interpolated = self._interpolate_pair(
                keyframes[i],
                keyframes[i + 1],
                num_interpolations,
                debug=debug
            )

            # Add all but the last frame (to avoid duplication)
            for j, frame in enumerate(interpolated[:-1]):
                all_frames.append(frame)

                if save_dir:
                    filename = f"frame_{frame_idx:04d}.png"
                    frame.save(save_path / filename)
                    frame_idx += 1
            
            # Clear CUDA cache periodically
            if i % 2 == 0:
                torch.cuda.empty_cache()

        # Add final keyframe
        all_frames.append(keyframes[-1])
        if save_dir:
            keyframes[-1].save(save_path / f"frame_{frame_idx:04d}.png")

        print(f"\n✓ Total frames: {len(all_frames)}")
        return all_frames

    def _interpolate_pair(self,
                         frame1: Image.Image,
                         frame2: Image.Image,
                         num_interpolations: int,
                         debug: bool = False) -> List[Image.Image]:
        """Interpolate between two frames in latent space"""

        frames = [frame1]

        # Ensure same size
        if frame1.size != frame2.size:
            print(f"⚠️  Resizing frame2 from {frame2.size} to {frame1.size}")
            frame2 = frame2.resize(frame1.size, Image.LANCZOS)

        # Encode to latent space
        latent1 = self.encode_image(frame1)
        latent2 = self.encode_image(frame2)
        
        if debug:
            print(f"  Latent1 shape: {latent1.shape}, range: [{latent1.min():.3f}, {latent1.max():.3f}]")
            print(f"  Latent2 shape: {latent2.shape}, range: [{latent2.min():.3f}, {latent2.max():.3f}]")

        # Generate interpolated frames
        for i in range(1, num_interpolations + 1):
            t = i / (num_interpolations + 1)

            # SLERP in latent space
            interpolated_latent = self.slerp(latent1, latent2, t)
            
            if debug:
                print(f"  Interp t={t:.3f}: range [{interpolated_latent.min():.3f}, {interpolated_latent.max():.3f}]")
            
            # Check for NaN or Inf
            if torch.isnan(interpolated_latent).any() or torch.isinf(interpolated_latent).any():
                print(f"⚠️  Warning: NaN/Inf detected in interpolated latent at t={t}")
                # Fallback to linear interpolation
                interpolated_latent = (1 - t) * latent1 + t * latent2

            # Decode back to image
            interpolated_image = self.decode_latent(interpolated_latent)
            
            if debug:
                # Check if image is black
                img_array = np.array(interpolated_image)
                print(f"  Decoded image: range [{img_array.min()}, {img_array.max()}], mean: {img_array.mean():.1f}")
            
            frames.append(interpolated_image)

        frames.append(frame2)
        return frames
    
    def test_encode_decode(self, image: Image.Image) -> Image.Image:
        """
        Test encode-decode round trip (for debugging)
        
        Args:
            image: Input image
            
        Returns:
            Reconstructed image
        """
        print("\n[TEST] Encode-Decode Round Trip")
        print(f"Input size: {image.size}")
        
        # Encode
        latent = self.encode_image(image)
        print(f"Latent shape: {latent.shape}")
        print(f"Latent range: [{latent.min():.3f}, {latent.max():.3f}]")
        print(f"Latent mean: {latent.mean():.3f}, std: {latent.std():.3f}")
        
        # Decode
        reconstructed = self.decode_latent(latent)
        print(f"Output size: {reconstructed.size}")
        
        # Check if black
        rec_array = np.array(reconstructed)
        print(f"Output range: [{rec_array.min()}, {rec_array.max()}]")
        print(f"Output mean: {rec_array.mean():.1f}")
        
        return reconstructed
    
    def unload(self):
        """Unload VAE from GPU to free memory"""
        self.vae.to('cpu')
        torch.cuda.empty_cache()
        print("✓ VAE unloaded from GPU")
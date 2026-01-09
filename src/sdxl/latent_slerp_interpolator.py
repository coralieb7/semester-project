"""
FILE: src/sdxl/latent_slerp_interpolator.py

Latent space SLERP interpolator for SDXL + norm deviation tracking
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Optional
from tqdm import tqdm
from pathlib import Path
from diffusers import AutoencoderKL
import time
import math


class LatentSlerpInterpolator:
    """Interpolate frames in SDXL latent space using SLERP, tracking norm deviation."""

    def __init__(self, vae: AutoencoderKL, device: str = "cuda"):
        self.vae = vae
        self.device = device

        # timings
        self.decode_times = []
        self.encode_times = []

        # norm deviation tracking (for the whole sequence)
        self.norm_dev_series = []  # list of dicts per interpolated latent

        # Keep VAE in float32 for stable encode/decode
        self.vae.to(device=device, dtype=torch.float32)
        self.vae.eval()

        print("✓ Latent SLERP Interpolator initialized (VAE in float32)")

    @staticmethod
    def _flat_norm(x: torch.Tensor, eps: float = 1e-8) -> float:
        return float(torch.norm(x.flatten()) + eps)

    @staticmethod
    def _norm_rel_dev(n_t: float, n0: float, n1: float, t: float, eps: float = 1e-8) -> float:
        n_lin = (1.0 - t) * n0 + t * n1
        return float(abs(n_t - n_lin) / (n_lin + eps))

    def slerp(self, v0: torch.Tensor, v1: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
        original_shape = v0.shape
        v0_flat = v0.flatten()
        v1_flat = v1.flatten()

        v0_norm = v0_flat / (torch.norm(v0_flat) + eps)
        v1_norm = v1_flat / (torch.norm(v1_flat) + eps)

        dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
        theta = torch.acos(dot)

        if theta.abs() < eps:
            result = (1.0 - t) * v0_flat + t * v1_flat
        else:
            sin_theta = torch.sin(theta)
            w0 = torch.sin((1.0 - t) * theta) / sin_theta
            w1 = torch.sin(t * theta) / sin_theta
            result = w0 * v0_flat + w1 * v1_flat

        return result.reshape(original_shape)

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(device=self.device, dtype=torch.float32)
        img_tensor = 2.0 * img_tensor - 1.0

        with torch.no_grad():
            t0 = time.perf_counter()
            torch.cuda.synchronize()
            latent_dist = self.vae.encode(img_tensor).latent_dist
            latent = latent_dist.sample()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.encode_times.append(t1 - t0)
            latent = latent * 0.13025  # SDXL scaling

        return latent

    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        with torch.no_grad():
            latent = latent / 0.13025

            t0 = time.perf_counter()
            torch.cuda.synchronize()
            decoded = self.vae.decode(latent).sample
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.decode_times.append(t1 - t0)


            decoded = (decoded + 1.0) / 2.0
            decoded = torch.clamp(decoded, 0.0, 1.0)

        decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
        decoded = (decoded * 255).astype(np.uint8)
        return Image.fromarray(decoded)

    def interpolate_between_keyframes(
        self,
        keyframes: List[Image.Image],
        num_interpolations: int = 5,
        save_dir: Optional[str] = None,
        debug: bool = False,
    ) -> List[Image.Image]:
        print("\n" + "=" * 70)
        print("Interpolating frames with Latent SLERP")
        print("=" * 70)
        print(f"Keyframes: {len(keyframes)}")
        print(f"Interpolations per pair: {num_interpolations}")
        print(f"VAE dtype: {next(self.vae.parameters()).dtype}")

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        all_frames = []
        frame_idx = 0

        for i in tqdm(range(len(keyframes) - 1), desc="Interpolating keyframe pairs"):
            interpolated = self._interpolate_pair(
                keyframes[i], keyframes[i + 1], num_interpolations, debug=debug
            )

            for frame in interpolated[:-1]:
                all_frames.append(frame)
                if save_dir:
                    frame.save(Path(save_dir) / f"frame_{frame_idx:04d}.png")
                    frame_idx += 1

            if i % 2 == 0:
                torch.cuda.empty_cache()

        all_frames.append(keyframes[-1])
        if save_dir:
            keyframes[-1].save(Path(save_dir) / f"frame_{frame_idx:04d}.png")

        print(f"\n✓ Total frames: {len(all_frames)}")
        return all_frames

    def _interpolate_pair(
        self,
        frame1: Image.Image,
        frame2: Image.Image,
        num_interpolations: int,
        debug: bool = False,
    ) -> List[Image.Image]:
        frames = [frame1]

        if frame1.size != frame2.size:
            frame2 = frame2.resize(frame1.size, Image.LANCZOS)

        latent1 = self.encode_image(frame1)
        latent2 = self.encode_image(frame2)

        # norms of endpoints (flattened)
        n0 = self._flat_norm(latent1)
        n1 = self._flat_norm(latent2)

        for i in range(1, num_interpolations + 1):
            t = i / (num_interpolations + 1)

            zt = self.slerp(latent1, latent2, t)

            if torch.isnan(zt).any() or torch.isinf(zt).any():
                # fallback
                zt = (1 - t) * latent1 + t * latent2

            # --- norm deviation logging ---
            n_t = self._flat_norm(zt)
            rel_dev = self._norm_rel_dev(n_t, n0, n1, t)
            self.norm_dev_series.append(
                {"pair": None, "t": float(t), "norm": float(n_t), "rel_dev": float(rel_dev)}
            )

            img = self.decode_latent(zt)
            frames.append(img)

        frames.append(frame2)
        return frames

    def get_decode_stats(self):
        if len(self.decode_times) == 0:
            return {"mean_decode_time": None, "std_decode_time": None}
        return {
            "mean_decode_time": float(np.mean(self.decode_times)),
            "std_decode_time": float(np.std(self.decode_times, ddof=1)) if len(self.decode_times) > 1 else 0.0,
        }
    def get_encode_stats(self):
        if len(self.decode_times) == 0:
            return {"mean_encode_time": None, "std_encode_time": None}
        return {
            "mean_encode_time": float(np.mean(self.encode_times)),
            "std_encode_time": float(np.std(self.encode_times, ddof=1)) if len(self.encode_times) > 1 else 0.0,
        }

    def get_norm_stats(self):
        """Sequence-level norm deviation summary (mean/std/max)."""
        if len(self.norm_dev_series) == 0:
            return {"norm_rel_dev_mean": None, "norm_rel_dev_std": None, "norm_rel_dev_max": None}
        vals = np.array([d["rel_dev"] for d in self.norm_dev_series], dtype=np.float32)
        return {
            "norm_rel_dev_mean": float(vals.mean()),
            "norm_rel_dev_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "norm_rel_dev_max": float(vals.max()),
        }

    def unload(self):
        self.vae.to("cpu")
        torch.cuda.empty_cache()
        print("✓ VAE unloaded from GPU")

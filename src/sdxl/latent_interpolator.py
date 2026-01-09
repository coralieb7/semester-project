"""
FILE: src/sdxl/interpolator.py

FrameInterpolator:
- default: pixel-space LERP/Cosine (fast, no VAE)
- optional: latent-space LERP/Cosine using SDXL VAE (comparable to SLERP)
"""

import numpy as np
from PIL import Image
from typing import List, Optional
from tqdm import tqdm
from pathlib import Path
import time

import torch
from diffusers import AutoencoderKL


class LatentInterpolator:
    """Interpolate frames between keyframes (pixel-space or latent-space)."""

    def __init__(
        self,
        interpolation_method: str = "linear",
        latent: bool = False,
        vae: Optional[AutoencoderKL] = None,
        device: str = "cuda",
    ):
        """
        Args:
            interpolation_method: "linear" or "cosine"
            latent: if True, interpolate in SDXL latent space using VAE
            vae: required if latent=True
            device: cuda/cpu
        """
        self.interpolation_method = interpolation_method
        self.latent = latent
        self.vae = vae
        self.device = device

        self.decode_times = []
        self.encode_times = []
        self.norm_dev_series = []

        if self.latent:
            if self.vae is None:
                raise ValueError("latent=True requires a VAE instance (sdxl.img2img_pipe.vae).")
            self.vae.to(device=self.device, dtype=torch.float32)
            self.vae.eval()

    @staticmethod
    def _alpha(method: str, i: int, num_interpolations: int) -> float:
        if method == "linear":
            return i / (num_interpolations + 1)
        elif method == "cosine":
            t = i / (num_interpolations + 1)
            return float((1 - np.cos(t * np.pi)) / 2)
        else:
            return i / (num_interpolations + 1)

    @staticmethod
    def _flat_norm(x: torch.Tensor, eps: float = 1e-8) -> float:
        return float(torch.norm(x.flatten()) + eps)

    @staticmethod
    def _norm_rel_dev(n_t: float, n0: float, n1: float, t: float, eps: float = 1e-8) -> float:
        n_lin = (1.0 - t) * n0 + t * n1
        return float(abs(n_t - n_lin) / (n_lin + eps))

    # ---------- VAE helpers (latent mode) ----------
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        img = np.array(image).astype(np.float32) / 255.0
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device, torch.float32)
        x = 2.0 * x - 1.0
        with torch.no_grad():
            t0 = time.perf_counter()
            torch.cuda.synchronize()
            z = self.vae.encode(x).latent_dist.sample()
            z = z * 0.13025
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.encode_times.append(t1 - t0)

        return z

    def _decode_latent(self, z: torch.Tensor) -> Image.Image:
        with torch.no_grad():
            z = z / 0.13025

            t0 = time.perf_counter()
            torch.cuda.synchronize()
            x = self.vae.decode(z).sample
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.decode_times.append(t1 - t0)

            x = (x + 1.0) / 2.0
            x = torch.clamp(x, 0.0, 1.0)

        x = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        x = (x * 255).astype(np.uint8)
        return Image.fromarray(x)

    # ---------- main API ----------
    def interpolate_between_keyframes(
        self,
        keyframes: List[Image.Image],
        num_interpolations: int = 5,
        save_dir: Optional[str] = None,
    ) -> List[Image.Image]:
        print(f"\n{'='*70}")
        print("Interpolating frames")
        print(f"{'='*70}")
        print(f"Keyframes: {len(keyframes)}")
        print(f"Interpolations per pair: {num_interpolations}")
        print(f"Method: {self.interpolation_method}")
        print(f"Space: {'latent' if self.latent else 'pixel'}")

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        all_frames = []
        frame_idx = 0

        for i in range(len(keyframes) - 1):
            interpolated = self._interpolate_pair(keyframes[i], keyframes[i + 1], num_interpolations)

            for frame in interpolated[:-1]:
                all_frames.append(frame)
                if save_dir:
                    frame.save(Path(save_dir) / f"frame_{frame_idx:04d}.png")
                    frame_idx += 1

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
    ) -> List[Image.Image]:
        frames = [frame1]

        if frame1.size != frame2.size:
            frame2 = frame2.resize(frame1.size, Image.LANCZOS)

        if not self.latent:
            # --- pixel interpolation (your original behavior) ---
            img1 = np.array(frame1).astype(np.float32) / 255.0
            img2 = np.array(frame2).astype(np.float32) / 255.0

            for i in range(1, num_interpolations + 1):
                alpha = self._alpha(self.interpolation_method, i, num_interpolations)
                out = (1 - alpha) * img1 + alpha * img2
                out = np.clip(out * 255, 0, 255).astype(np.uint8)
                frames.append(Image.fromarray(out))

            frames.append(frame2)
            return frames

        # --- latent interpolation ---
        z1 = self._encode_image(frame1)
        z2 = self._encode_image(frame2)

        n0 = self._flat_norm(z1)
        n1 = self._flat_norm(z2)

        for i in range(1, num_interpolations + 1):
            t = i / (num_interpolations + 1)
            alpha = self._alpha(self.interpolation_method, i, num_interpolations)

            zt = (1 - alpha) * z1 + alpha * z2

            # norm deviation logging
            n_t = self._flat_norm(zt)
            rel_dev = self._norm_rel_dev(n_t, n0, n1, t)
            self.norm_dev_series.append({"t": float(t), "norm": float(n_t), "rel_dev": float(rel_dev)})

            img = self._decode_latent(zt)
            frames.append(img)

        frames.append(frame2)
        return frames

    def get_decode_stats(self):
        if not self.latent or len(self.decode_times) == 0:
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
        if not self.latent or len(self.norm_dev_series) == 0:
            return {"norm_rel_dev_mean": None, "norm_rel_dev_std": None, "norm_rel_dev_max": None}
        vals = np.array([d["rel_dev"] for d in self.norm_dev_series], dtype=np.float32)
        return {
            "norm_rel_dev_mean": float(vals.mean()),
            "norm_rel_dev_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "norm_rel_dev_max": float(vals.max()),
        }

    def unload(self):
        if self.latent and self.vae is not None:
            self.vae.to("cpu")
        torch.cuda.empty_cache()
        print("✓ Interpolator unloaded")

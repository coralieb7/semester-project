"""
FILE: src/sdxl/interpolator.py
"""

import numpy as np
from PIL import Image
from typing import List, Optional
from tqdm import tqdm
from pathlib import Path


class FrameInterpolator:
    """Interpolate frames between keyframes"""
    
    def __init__(self, interpolation_method: str = "linear"):
        """
        Initialize interpolator
        
        Args:
            interpolation_method: Method for interpolation ("linear", "cosine")
        """
        self.interpolation_method = interpolation_method
    
    def interpolate_between_keyframes(self,
                                     keyframes: List[Image.Image],
                                     num_interpolations: int = 5,
                                     save_dir: Optional[str] = None) -> List[Image.Image]:
        """
        Interpolate frames between all keyframes
        
        Args:
            keyframes: List of keyframe images
            num_interpolations: Number of frames to add between each pair
            save_dir: Optional directory to save interpolated frames
            
        Returns:
            List of all frames (keyframes + interpolated)
        """
        print(f"\n{'='*70}")
        print(f"Interpolating frames")
        print(f"{'='*70}")
        print(f"Keyframes: {len(keyframes)}")
        print(f"Interpolations per pair: {num_interpolations}")
        print(f"Method: {self.interpolation_method}")
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        all_frames = []
        frame_idx = 0
        
        # Process each pair of keyframes
        for i in range(len(keyframes) - 1):
            print(f"\nInterpolating between keyframe {i} and {i+1}...")
            
            interpolated = self._interpolate_pair(
                keyframes[i],
                keyframes[i + 1],
                num_interpolations
            )
            
            # Add all but the last frame (to avoid duplication)
            for frame in interpolated[:-1]:
                all_frames.append(frame)
                
                if save_dir:
                    filename = f"frame_{frame_idx:04d}.png"
                    frame.save(save_path / filename)
                    frame_idx += 1
        
        # Add final keyframe
        all_frames.append(keyframes[-1])
        if save_dir:
            keyframes[-1].save(save_path / f"frame_{frame_idx:04d}.png")
        
        print(f"\n✓ Total frames: {len(all_frames)}")
        return all_frames
    
    def _interpolate_pair(self,
                         frame1: Image.Image,
                         frame2: Image.Image,
                         num_interpolations: int) -> List[Image.Image]:
        """Interpolate between two frames"""
        
        frames = [frame1]
        
        # Ensure same size
        if frame1.size != frame2.size:
            frame2 = frame2.resize(frame1.size, Image.LANCZOS)
        
        # Convert to numpy arrays
        img1 = np.array(frame1).astype(np.float32) / 255.0
        img2 = np.array(frame2).astype(np.float32) / 255.0
        
        # Generate interpolated frames
        for i in range(1, num_interpolations + 1):
            # Calculate interpolation weight
            if self.interpolation_method == "linear":
                alpha = i / (num_interpolations + 1)
            elif self.interpolation_method == "cosine":
                # Cosine interpolation for smoother easing
                t = i / (num_interpolations + 1)
                alpha = (1 - np.cos(t * np.pi)) / 2
            else:
                alpha = i / (num_interpolations + 1)
            
            # Interpolate
            interpolated = (1 - alpha) * img1 + alpha * img2
            interpolated = np.clip(interpolated * 255, 0, 255).astype(np.uint8)
            
            frames.append(Image.fromarray(interpolated))
        
        frames.append(frame2)
        return frames

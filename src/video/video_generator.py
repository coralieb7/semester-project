"""
FILE: src/video/video_generator.py
"""

import subprocess
import numpy as np
from PIL import Image
from typing import List, Callable, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import tempfile
import shutil


class VideoGenerator:
    """Generate videos from frames using FFmpeg with signal mapping"""
    
    def __init__(self):
        """Initialize video generator"""
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE,
                          check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "FFmpeg not found! Please install FFmpeg:\n"
                "Ubuntu: sudo apt install ffmpeg\n"
                "Windows: Download from https://ffmpeg.org/"
            )
    
    def create_video_from_signal(self,
                                 all_frames: List[Image.Image],
                                 signal_function: Callable[[float], float],
                                 output_path: str,
                                 duration_seconds: float = 10.0,
                                 fps: int = 30,
                                 codec: str = 'libx264',
                                 quality: str = 'high') -> None:
        """
        Create video by sampling frames according to a signal function
        
        Args:
            all_frames: List of all available frames (mapped 0.0 to 1.0)
            signal_function: Function that maps time [0,1] to frame position [0,1]
            output_path: Output video path
            duration_seconds: Video duration in seconds
            fps: Frames per second
            codec: Video codec (libx264, libx265, etc.)
            quality: Quality preset (low, medium, high, veryhigh)
        """
        
        if not all_frames:
            raise ValueError("No frames provided")
        
        num_frames = len(all_frames)
        total_output_frames = int(duration_seconds * fps)
        
        print(f"\nVideo Generation Configuration:")
        print(f"  Input frames: {num_frames}")
        print(f"  Output frames: {total_output_frames}")
        print(f"  Duration: {duration_seconds}s @ {fps} FPS")
        print(f"  Codec: {codec}")
        print(f"  Quality: {quality}")
        
        # Create temporary directory for frame sequence
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            print("\nSampling frames according to signal...")
            sampled_frames = []
            
            for i in tqdm(range(total_output_frames), desc="Sampling frames"):
                # Time normalized to [0, 1]
                t = i / (total_output_frames - 1) if total_output_frames > 1 else 0.0
                
                # Get signal value (should be in [0, 1])
                signal_value = signal_function(t)
                signal_value = np.clip(signal_value, 0.0, 1.0)
                
                # Map signal to frame index
                frame_idx = int(signal_value * (num_frames - 1))
                frame_idx = np.clip(frame_idx, 0, num_frames - 1)
                
                sampled_frames.append(all_frames[frame_idx])
            
            # Save sampled frames as sequence
            print("\nWriting frame sequence...")
            for i, frame in enumerate(tqdm(sampled_frames, desc="Writing frames")):
                frame_path = temp_path / f"frame_{i:06d}.png"
                frame.save(frame_path)
            
            # FFmpeg quality settings
            quality_presets = {
                'low': {'crf': '28', 'preset': 'fast'},
                'medium': {'crf': '23', 'preset': 'medium'},
                'high': {'crf': '18', 'preset': 'slow'},
                'veryhigh': {'crf': '15', 'preset': 'slower'}
            }
            
            settings = quality_presets.get(quality, quality_presets['high'])
            
            # Build FFmpeg command
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-framerate', str(fps),
                '-i', str(temp_path / 'frame_%06d.png'),
                '-c:v', codec,
                '-pix_fmt', 'yuv420p',
            ]
            
            # Add codec-specific settings
            if codec in ['libx264', 'libx265']:
                ffmpeg_cmd.extend([
                    '-crf', settings['crf'],
                    '-preset', settings['preset']
                ])
            
            ffmpeg_cmd.append(str(output_path))
            
            # Run FFmpeg
            print("\nEncoding video with FFmpeg...")
            try:
                result = subprocess.run(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True
                )
                print(f"✓ Video created successfully: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"\nFFmpeg Error:")
                print(e.stderr)
                raise RuntimeError(f"FFmpeg failed: {e}")
    
    @staticmethod
    def visualize_signal(signal_function: Callable[[float], float],
                        num_samples: int = 1000,
                        output_path: Optional[str] = None) -> None:
        """
        Visualize a signal function
        
        Args:
            signal_function: Signal function to visualize
            num_samples: Number of samples
            output_path: Path to save plot (optional)
        """
        t = np.linspace(0, 1, num_samples)
        signal_values = np.array([signal_function(ti) for ti in t])
        
        plt.figure(figsize=(12, 6))
        
        # Signal plot
        plt.subplot(2, 1, 1)
        plt.plot(t, signal_values, linewidth=2)
        plt.xlabel('Normalized Time')
        plt.ylabel('Signal Value')
        plt.title('Signal Function')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        
        # Frame index plot
        plt.subplot(2, 1, 2)
        frame_indices = (signal_values * 100).astype(int)  # Assuming 100 frames
        plt.plot(t, frame_indices, linewidth=2, color='orange')
        plt.xlabel('Normalized Time')
        plt.ylabel('Frame Index')
        plt.title('Frame Selection Over Time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Signal visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_comparison_video(self,
                               all_frames: List[Image.Image],
                               signal_functions: List[tuple],  # [(name, function), ...]
                               output_dir: str,
                               duration_seconds: float = 10.0,
                               fps: int = 30) -> None:
        """
        Create multiple videos with different signal functions for comparison
        
        Args:
            all_frames: List of all frames
            signal_functions: List of (name, function) tuples
            output_dir: Output directory
            duration_seconds: Video duration
            fps: Frames per second
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCreating {len(signal_functions)} comparison videos...")
        
        for name, signal_func in signal_functions:
            print(f"\n{'='*70}")
            print(f"Creating video: {name}")
            print(f"{'='*70}")
            
            video_path = output_path / f"video_{name}.mp4"
            
            self.create_video_from_signal(
                all_frames=all_frames,
                signal_function=signal_func,
                output_path=str(video_path),
                duration_seconds=duration_seconds,
                fps=fps
            )
            
            # Also save signal visualization
            plot_path = output_path / f"signal_{name}.png"
            self.visualize_signal(signal_func, output_path=str(plot_path))
        
        print(f"\n✓ All comparison videos created in: {output_dir}")
    
    def concatenate_videos(self,
                          video_paths: List[str],
                          output_path: str,
                          orientation: str = 'horizontal') -> None:
        """
        Concatenate multiple videos side by side or vertically
        
        Args:
            video_paths: List of video paths
            output_path: Output path
            orientation: 'horizontal' or 'vertical'
        """
        if len(video_paths) < 2:
            raise ValueError("Need at least 2 videos to concatenate")
        
        # Build FFmpeg filter
        if orientation == 'horizontal':
            filter_complex = ''.join([f'[{i}:v]' for i in range(len(video_paths))])
            filter_complex += f'hstack=inputs={len(video_paths)}[v]'
        else:  # vertical
            filter_complex = ''.join([f'[{i}:v]' for i in range(len(video_paths))])
            filter_complex += f'vstack=inputs={len(video_paths)}[v]'
        
        # Build command
        cmd = ['ffmpeg', '-y']
        for video_path in video_paths:
            cmd.extend(['-i', video_path])
        
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[v]',
            '-c:v', 'libx264',
            '-crf', '18',
            output_path
        ])
        
        print(f"\nConcatenating {len(video_paths)} videos ({orientation})...")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Concatenated video saved: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            raise


"""
FILE: scripts/3_create_video.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from PIL import Image
from src.video.video_generator import VideoGenerator
from src.sdxl.mapper import SignalMapper, SignalFunctions
from config.paths import Paths
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Create video from generated frames using signal"
    )
    
    # Input
    parser.add_argument('--project', type=str, required=True,
                       help='Project directory name (in data/output/)')
    
    # Signal parameters
    parser.add_argument('--signal', type=str, default='linear',
                       choices=['linear', 'reverse', 'sine', 'cosine', 'triangle',
                               'sawtooth', 'square', 'ease', 'bounce', 'custom'],
                       help='Signal function type')
    parser.add_argument('--frequency', type=float, default=1.0,
                       help='Signal frequency (for periodic signals)')
    parser.add_argument('--phase', type=float, default=0.0,
                       help='Signal phase offset (for periodic signals)')
    parser.add_argument('--custom-expr', type=str, default=None,
                       help='Custom signal expression (e.g., "sin(2*pi*t)")')
    
    # Video parameters
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Video duration in seconds')
    parser.add_argument('--fps', type=int, default=30,
                       help='Output frames per second')
    parser.add_argument('--codec', type=str, default='libx264',
                       help='Video codec')
    parser.add_argument('--quality', type=str, default='high',
                       choices=['low', 'medium', 'high', 'veryhigh'],
                       help='Video quality preset')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output video filename')
    parser.add_argument('--visualize-signal', action='store_true',
                       help='Save signal visualization plot')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Video Creation from Signal")
    print("="*70)
    
    # Load project
    project_dir = Paths.OUTPUT_DIR / args.project
    if not project_dir.exists():
        print(f"ERROR: Project not found: {project_dir}")
        sys.exit(1)
    
    print(f"\nProject: {args.project}")
    print(f"Location: {project_dir}")
    
    # Load metadata
    metadata_file = project_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"\nMetadata loaded:")
        print(f"  Total frames: {metadata['total_frames']}")
        print(f"  Keyframes: {metadata['num_keyframes']}")
    else:
        print("\nWarning: metadata.json not found")
        metadata = None
    
    # Load all frames
    frames_dir = project_dir / "frames"
    print(f"\nLoading frames from: {frames_dir}")
    
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        print("ERROR: No frames found!")
        sys.exit(1)
    
    print(f"Loading {len(frame_files)} frames...")
    all_frames = [Image.open(f) for f in frame_files]
    print(f"✓ Loaded {len(all_frames)} frames")
    
    # Create signal function
    print(f"\n{'='*70}")
    print(f"Signal Configuration")
    print(f"{'='*70}")
    print(f"Type: {args.signal}")
    
    if args.signal == 'linear':
        signal_func = SignalFunctions.linear
    elif args.signal == 'reverse':
        signal_func = SignalFunctions.reverse_linear
    elif args.signal == 'sine':
        signal_func = lambda t: SignalFunctions.sine(t, args.frequency, args.phase)
        print(f"Frequency: {args.frequency} Hz")
        print(f"Phase: {args.phase} rad")
    elif args.signal == 'cosine':
        signal_func = lambda t: SignalFunctions.cosine(t, args.frequency, args.phase)
        print(f"Frequency: {args.frequency} Hz")
        print(f"Phase: {args.phase} rad")
    elif args.signal == 'triangle':
        signal_func = lambda t: SignalFunctions.triangle(t, args.frequency)
        print(f"Frequency: {args.frequency} Hz")
    elif args.signal == 'sawtooth':
        signal_func = lambda t: SignalFunctions.sawtooth(t, args.frequency)
        print(f"Frequency: {args.frequency} Hz")
    elif args.signal == 'square':
        signal_func = lambda t: SignalFunctions.square(t, args.frequency)
        print(f"Frequency: {args.frequency} Hz")
    elif args.signal == 'ease':
        signal_func = SignalFunctions.ease_in_out
    elif args.signal == 'bounce':
        signal_func = lambda t: SignalFunctions.bounce(t, num_bounces=3)
    elif args.signal == 'custom':
        if not args.custom_expr:
            print("ERROR: --custom-expr required for custom signal")
            sys.exit(1)
        signal_func = lambda t: SignalFunctions.custom_function(t, args.custom_expr)
        print(f"Expression: {args.custom_expr}")
    
    # Visualize signal if requested
    if args.visualize_signal:
        signal_plot_path = project_dir / f"signal_{args.signal}.png"
        VideoGenerator.visualize_signal(
            signal_func,
            num_samples=1000,
            output_path=str(signal_plot_path)
        )
    
    # Create video
    print(f"\n{'='*70}")
    print(f"Creating Video")
    print(f"{'='*70}")
    print(f"Duration: {args.duration}s @ {args.fps} FPS")
    print(f"Total output frames: {int(args.duration * args.fps)}")
    
    output_filename = args.output or f"video_{args.signal}.mp4"
    output_path = project_dir / output_filename
    
    video_gen = VideoGenerator()
    
    video_gen.create_video_from_signal(
        all_frames=all_frames,
        signal_function=signal_func,
        output_path=str(output_path),
        duration_seconds=args.duration,
        fps=args.fps,
        codec=args.codec,
        quality=args.quality
    )
    
    print("\n" + "="*70)
    print("✓ VIDEO CREATION COMPLETE")
    print("="*70)
    print(f"\nVideo saved to: {output_path}")
    print(f"Duration: {args.duration}s")
    print(f"FPS: {args.fps}")
    print(f"Signal: {args.signal}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

"""
FILE: src/sdxl/mapper.py
"""

from typing import List, Callable
import numpy as np
from pathlib import Path
import json


class SignalMapper:
    """Map images to signal values (0.0 to 1.0)"""
    
    def __init__(self, num_frames: int):
        """
        Initialize signal mapper
        
        Args:
            num_frames: Total number of frames
        """
        self.num_frames = num_frames
        self.mappings = self._create_linear_mapping()
        
        print(f"✓ Signal Mapper initialized for {num_frames} frames")
    
    def _create_linear_mapping(self) -> List[float]:
        """Create linear mapping from 0.0 to 1.0"""
        if self.num_frames == 1:
            return [0.5]
        return [i / (self.num_frames - 1) for i in range(self.num_frames)]
    
    def get_frame_for_signal(self, signal_value: float) -> int:
        """
        Get frame index for a given signal value
        
        Args:
            signal_value: Signal value between 0.0 and 1.0
            
        Returns:
            Frame index
        """
        signal_value = np.clip(signal_value, 0.0, 1.0)
        
        # Find closest frame
        distances = [abs(signal_value - mapping) for mapping in self.mappings]
        return distances.index(min(distances))
    
    def get_signal_for_frame(self, frame_idx: int) -> float:
        """Get signal value for a frame index"""
        return self.mappings[frame_idx]
    
    def create_signal_sequence(self, 
                              signal_function: Callable[[float], float],
                              duration_seconds: float,
                              fps: int = 30) -> List[int]:
        """
        Create frame sequence based on a signal function
        
        Args:
            signal_function: Function that takes time (0-1) and returns signal (0-1)
            duration_seconds: Duration of output video
            fps: Frames per second
            
        Returns:
            List of frame indices
        """
        num_output_frames = int(duration_seconds * fps)
        frame_sequence = []
        
        for i in range(num_output_frames):
            t = i / (num_output_frames - 1) if num_output_frames > 1 else 0.0
            signal = signal_function(t)
            frame_idx = self.get_frame_for_signal(signal)
            frame_sequence.append(frame_idx)
        
        return frame_sequence
    
    def save_mapping(self, output_path: str):
        """Save signal mapping to file"""
        
        mapping_data = {
            'num_frames': self.num_frames,
            'mappings': self.mappings
        }
        
        with open(output_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f"✓ Mapping saved to: {output_path}")
    
    @staticmethod
    def load_mapping(input_path: str) -> 'SignalMapper':
        """Load signal mapping from file"""
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        mapper = SignalMapper(data['num_frames'])
        mapper.mappings = data['mappings']
        
        print(f"✓ Mapping loaded from: {input_path}")
        return mapper


# Predefined signal functions
class SignalFunctions:
    """Common signal functions for video generation"""
    
    @staticmethod
    def linear(t: float) -> float:
        """Linear progression from 0 to 1"""
        return t
    
    @staticmethod
    def reverse_linear(t: float) -> float:
        """Linear progression from 1 to 0"""
        return 1.0 - t
    
    @staticmethod
    def sine(t: float, frequency: float = 1.0, phase: float = 0.0) -> float:
        """Sinusoidal signal"""
        return (np.sin(2 * np.pi * frequency * t + phase) + 1) / 2
    
    @staticmethod
    def cosine(t: float, frequency: float = 1.0, phase: float = 0.0) -> float:
        """Cosinusoidal signal"""
        return (np.cos(2 * np.pi * frequency * t + phase) + 1) / 2
    
    @staticmethod
    def triangle(t: float, frequency: float = 1.0) -> float:
        """Triangle wave"""
        t_mod = (t * frequency) % 1.0
        return 2 * abs(t_mod - 0.5)
    
    @staticmethod
    def sawtooth(t: float, frequency: float = 1.0) -> float:
        """Sawtooth wave"""
        return (t * frequency) % 1.0
    
    @staticmethod
    def square(t: float, frequency: float = 1.0, duty_cycle: float = 0.5) -> float:
        """Square wave"""
        t_mod = (t * frequency) % 1.0
        return 1.0 if t_mod < duty_cycle else 0.0
    
    @staticmethod
    def ease_in_out(t: float) -> float:
        """Smooth ease in/out (cosine interpolation)"""
        return (1 - np.cos(t * np.pi)) / 2
    
    @staticmethod
    def bounce(t: float, num_bounces: int = 3) -> float:
        """Bouncing signal"""
        return abs(np.sin(num_bounces * np.pi * t))
    
    @staticmethod
    def custom_function(t: float, expression: str) -> float:
        """
        Custom mathematical expression
        Available: t, sin, cos, tan, exp, log, sqrt, abs
        Example: "sin(2*pi*t) * exp(-t)"
        """
        import math
        pi = math.pi
        sin = math.sin
        cos = math.cos
        tan = math.tan
        exp = math.exp
        log = math.log
        sqrt = math.sqrt
        
        try:
            result = eval(expression)
            return np.clip(result, 0.0, 1.0)
        except:
            return t

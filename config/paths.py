"""
FILE: config/paths.py
"""

from pathlib import Path


class Paths:
    """Central path management"""
    
    # Base paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    OUTPUT_DIR = DATA_DIR / "output"
    
    # Data paths
    INPUT_DIR = DATA_DIR / "input"
    GUIDELINES_DIR = DATA_DIR / "guidelines"
    TRAINING_DATA_DIR = DATA_DIR / "training_data"
    
    # Output paths
    FRAMES_DIR = OUTPUT_DIR / "frames"
    VIDEOS_DIR = OUTPUT_DIR / "videos"
    PROMPTS_DIR = OUTPUT_DIR / "prompts"
    
    # Model paths
    LORA_WEIGHTS_DIR = MODELS_DIR / "lora_weights"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    CHROMA_DB_DIR = MODELS_DIR / "chroma_db"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.INPUT_DIR,
            cls.GUIDELINES_DIR,
            cls.TRAINING_DATA_DIR,
            cls.OUTPUT_DIR,
            cls.FRAMES_DIR,
            cls.VIDEOS_DIR,
            cls.PROMPTS_DIR,
            cls.MODELS_DIR,
            cls.LORA_WEIGHTS_DIR,
            cls.CHECKPOINTS_DIR,
            cls.CHROMA_DB_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✓ All directories created")

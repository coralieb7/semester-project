"""
Configuration and Documentation
================================
FILE: config/model_config.py
"""

class ModelConfig:
    """Central configuration for all models"""
    
    # VLM Configuration (Qwen2.5-VL)
    VLM_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    VLM_DEVICE = "cuda"
    VLM_DTYPE = "auto"
    
    # SDXL Configuration
    SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    SDXL_DEVICE = "cuda"
    SDXL_DTYPE = "float16"
    
    # RAG Configuration
    RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RAG_COLLECTION_NAME = "visual_guidelines"
    RAG_PERSIST_DIR = "./models/chroma_db"
    
    # LoRA Configuration
    LORA_RANK = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    
    # Generation Defaults
    DEFAULT_NUM_PROMPTS = 13
    DEFAULT_INTERPOLATIONS = 5
    DEFAULT_STRENGTH = 0.25
    DEFAULT_GUIDANCE_SCALE = 15.0
    DEFAULT_NUM_INFERENCE_STEPS = 40
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    
    # Video Defaults
    DEFAULT_FPS = 30
    DEFAULT_DURATION = 10.0
    DEFAULT_CODEC = "libx264"
    DEFAULT_QUALITY = "high"

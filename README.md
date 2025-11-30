# SDXL Video Generation with VLM Signal Mapping

Advanced video generation system using:
- **Qwen2.5-VL**: Vision-language model for evolutionary prompt generation
- **SDXL**: High-quality image generation with LoRA support
- **Signal Mapping**: Map frames to signals (0.0-1.0) for dynamic video creation
- **FFmpeg**: Professional video encoding with signal-based frame selection

## Core Concept

### Reverse Engineering Approach

1. **VLM Analysis**: Analyze a target image and generate N evolutionary prompts
2. **Image Generation**: Use SDXL img2img to generate keyframes from prompts
3. **Interpolation**: Create smooth transitions between keyframes
4. **Signal Mapping**: Map all frames to normalized signal values (0.0 to 1.0)
5. **Video Creation**: Generate videos using signal functions (sine, linear, etc.)

### Signal Mapping

Every frame is mapped to a decimal value between 0.0 and 1.0:
- 100 frames → frame 0 = 0.00, frame 1 = 0.01, ..., frame 99 = 1.00
- Create videos by sampling frames based on signal functions
- Example: Sinusoidal signal creates oscillating forward/backward motion

## Quick Start

### Installation

```bash
# Install FFmpeg
sudo apt install ffmpeg

# Install Python dependencies
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install diffusers transformers accelerate peft qwen-vl-utils
pip install sentence-transformers chromadb pillow opencv-python
pip install pypdf2 python-docx tqdm

# Create project structure
python -c "from config.paths import Paths; Paths.create_directories()"
```

### Basic Workflow

```bash
# Step 1: Generate images with evolutionary prompts
python scripts/2_generate_images.py \\
    --initial-image data/input/neuron.png \\
    --evolution "the dot becomes a neuron with multidirectional branches" \\
    --num-prompts 13 \\
    --interpolations 5

# Step 2: Create video with signal
python scripts/3_create_video.py \\
    --project generation_20241124_153045 \\
    --signal sine \\
    --frequency 1.0 \\
    --duration 10 \\
    --fps 30
```

## Detailed Usage

### 1. Generate Evolutionary Images

```bash
python scripts/2_generate_images.py \\
    --initial-image data/input/target.png \\
    --evolution "description of evolution" \\
    --num-prompts 13 \\
    --num-keyframes 13 \\
    --analysis-type "technical" \\
    --interpolations 5 \\
    --strength 0.25 \\
    --guidance 15.0 \\
    --steps 40
```

**Options:**
- `--initial-image`: Target/reference image for reverse engineering
- `--evolution`: Description of evolution (e.g., "dot to complex network")
- `--num-prompts`: Max sequence length (default: 13)
- `--num-keyframes`: How many images to generate (default: same as prompts)
- `--analysis-type`: What type of description for the initial image (default: detailed)
- `--interpolations`: Frames between keyframes for smoothness (default: 5)
- `--strength`: How much each frame changes (0.15-0.35)
- `--use-txt2img`: Start with txt2img instead of initial image
- `--txt2img-prompt`: Custom prompt for txt2img
- `--lora`: Path to LoRA weights
- `--use-rag`: Enable RAG guideline enhancement

### 2. Create Signal-Based Videos

```bash
# Linear progression (0 to 1)
python scripts/3_create_video.py \\
    --project your_project_name \\
    --signal linear \\
    --duration 10 \\
    --fps 30

# Sinusoidal oscillation
python scripts/3_create_video.py \\
    --project your_project_name \\
    --signal sine \\
    --frequency 2.0 \\
    --duration 15 \\
    --fps 30

# Custom mathematical expression
python scripts/3_create_video.py \\
    --project your_project_name \\
    --signal custom \\
    --custom-expr "sin(2*pi*t) * exp(-t)" \\
    --duration 10
```

**Available Signals:**
- `linear`: Straight progression 0→1
- `reverse`: Reverse progression 1→0
- `sine`: Sinusoidal wave (oscillating)
- `cosine`: Cosine wave
- `triangle`: Triangle wave (linear up/down)
- `sawtooth`: Sawtooth wave (linear up, instant reset)
- `square`: Square wave (binary on/off)
- `ease`: Smooth ease in/out
- `bounce`: Bouncing effect
- `custom`: Custom mathematical expression

### 3. Train Custom LoRA

#### 1. Create template pairing.json from your 105 images
```bash
python scripts/create_pairing_template.py --image-dir data/training_data
```

#### 2. Edit data/training_data/pairing.json with proper prompts

#### 3. Train LoRA
```bash
python scripts/1_train_lora.py \
    --dataset-dir data/training_data \
    --epochs 15 \
    --lora-rank 16 \
    --save-every 5
```
#### 4. Use trained LoRA
```bash
python scripts/2_generate_images.py \
    --initial-image data/input/dot.png \
    --evolution "dot becomes complex form" \
    --lora models/lora_weights/custom_lora/checkpoint_epoch_15
```

### 4. Debug VLM

```bash
# Analyze image
python scripts/debug_vlm.py \\
    --image data/input/test.png \\
    --mode analyze \\
    --analysis-type detailed

# Extract features
python scripts/debug_vlm.py \\
    --image data/input/test.png \\
    --mode features

# Compare two images
python scripts/debug_vlm.py \\
    --image data/input/img1.png \\
    --image2 data/input/img2.png \\
    --mode compare

# Test prompt generation
python scripts/debug_vlm.py \\
    --image data/input/test.png \\
    --mode prompts \\
    --evolution "becomes more complex" \\
    --num-prompts 5 \\
    --save prompts_debug.json
```

## Project Structure

```
sdxl_video_project/
├── config/
│   ├── model_config.py       # Model configurations
│   └── paths.py              # Path management
│
├── data/
│   ├── input/                # Input images
│   ├── guidelines/           # Visual guidelines (for RAG)
│   ├── training_data/        # LoRA training data
│   └── output/               # Generated outputs
│       └── project_name/
│           ├── frames/       # All generated frames
│           ├── prompts.json  # Generated prompts
│           ├── signal_mapping.json
│           ├── metadata.json
│           └── video_*.mp4   # Generated videos
│
├── models/
│   ├── lora_weights/         # Trained LoRA models
│   ├── checkpoints/          # Training checkpoints
│   └── chroma_db/            # RAG vector database
│
├── src/
│   ├── vlm/
│   │   ├── image_analyzer.py     # Image analysis
│   │   └── prompt_generator.py   # Evolutionary prompts
│   │
│   ├── sdxl/
│   │   ├── generator.py          # SDXL generation
│   │   ├── interpolator.py       # Frame interpolation
│   │   ├── mapper.py             # Signal mapping
│   │   └── lora_trainer.py       # LoRA training
│   │
│   ├── video/
│   │   ├── video_generator.py    # FFmpeg video creation
│   │   └── effects.py            # Video effects
│   │
│   └── rag/
│       └── guideline_rag.py      # RAG system (later)
│
└── scripts/
    ├── 1_train_lora.py           # Train LoRA
    ├── 2_generate_images.py      # Generate evolutionary images
    ├── 3_create_video.py         # Create signal-based video
    └── debug_vlm.py              # VLM debugging tool
```

## Example Workflows

### Workflow 1: Simple Evolution Video

```bash
# Generate images
python scripts/2_generate_images.py \\
    --initial-image data/input/neuron.png \\
    --evolution "single neuron to complex neural network" \\
    --num-prompts 10 \\
    --interpolations 7

# Create linear video
python scripts/3_create_video.py \\
    --project generation_* \\
    --signal linear \\
    --duration 12 \\
    --fps 30
```

### Workflow 2: With LoRA and Custom Signal

```bash
# Train LoRA on your style
python scripts/1_train_lora.py \\
    --dataset data/training_data/scientific_style \\
    --output models/lora_weights/scientific \\
    --epochs 20

# Generate with LoRA
python scripts/2_generate_images.py \\
    --initial-image data/input/cell.png \\
    --evolution "cell division process" \\
    --lora models/lora_weights/scientific \\
    --num-prompts 15

# Create video with sinusoidal motion
python scripts/3_create_video.py \\
    --project generation_* \\
    --signal sine \\
    --frequency 0.5 \\
    --duration 20 \\
    --visualize-signal
```

### Workflow 3: Start from Text (no initial image)

```bash
# Generate starting with txt2img
python scripts/2_generate_images.py \\
    --initial-image data/input/reference.png \\
    --use-txt2img \\
    --txt2img-prompt "a simple geometric shape, minimal" \\
    --evolution "shape evolves into a complex fractal pattern" \\
    --num-prompts 20

# Create bounce effect video
python scripts/3_create_video.py \\
    --project generation_* \\
    --signal bounce \\
    --duration 8
```

## Advanced Features

### Signal Mapping System

The signal mapper creates a bijection between frames and signal values:

```python
from src.sdxl.mapper import SignalMapper, SignalFunctions

# Create mapper
mapper = SignalMapper(num_frames=100)

# Get frame for signal value
frame_idx = mapper.get_frame_for_signal(0.75)  # Returns frame 75

# Get signal for frame
signal = mapper.get_signal_for_frame(50)  # Returns 0.50

# Create custom signal sequence
def custom_signal(t):
    return (np.sin(4 * np.pi * t) + 1) / 2

sequence = mapper.create_signal_sequence(
    signal_function=custom_signal,
    duration_seconds=10,
    fps=30
)
```

### Custom Signal Functions

```python
from src.sdxl.mapper import SignalFunctions
import numpy as np

# Combine signals
def combined_signal(t):
    sine = SignalFunctions.sine(t, frequency=2.0)
    ease = SignalFunctions.ease_in_out(t)
    return 0.7 * sine + 0.3 * ease

# Complex mathematical expression
python scripts/3_create_video.py \\
    --project my_project \\
    --signal custom \\
    --custom-expr "0.5 + 0.5*sin(4*pi*t) * exp(-2*t)"
```

## Configuration

Edit `config/model_config.py`:

```python
DEFAULT_NUM_PROMPTS = 13          # Max sequence length
DEFAULT_INTERPOLATIONS = 5        # Smoothness
DEFAULT_STRENGTH = 0.25          # SDXL change amount
DEFAULT_GUIDANCE_SCALE = 15.0    # Prompt adherence
DEFAULT_FPS = 30                 # Video frame rate
```

## Tips

### For Smooth Videos
- Use low strength (0.20-0.30)
- More interpolations (7-10)
- Gradual prompt changes
- More keyframes with smaller evolutionary steps

### For Signal-Based Videos
- `sine` and `cosine` create oscillating motion
- `triangle` creates smooth forward-backward loops
- `ease` creates cinematic slow-start/slow-end
- Combine signals for complex motion

### For Best Quality
- Use high inference steps (50+)
- Higher guidance scale (15-20) for consistency
- Train LoRA for specific style consistency
- Use RAG for guideline adherence

## Troubleshooting

**Out of Memory:**
- Reduce image resolution (--width 768 --height 768)
- Reduce batch size for LoRA training
- Lower inference steps

**Inconsistent Results:**
- Increase guidance scale (18-20)
- Reduce strength (0.15-0.20)
- Use LoRA for style consistency
- More keyframes with smaller changes

**FFmpeg Not Found:**
```bash
sudo apt install ffmpeg  # Ubuntu
brew install ffmpeg      # macOS
```

## Performance

**RTX 4090 (24GB VRAM):**
- VLM prompt generation: ~10-15s
- SDXL keyframe: ~4-6s per image
- Frame interpolation: ~0.1s per frame
- FFmpeg encoding: ~5-10s for 30s video

**Total for 13 keyframes + 5 interpolations:**
- ~100 frames generated
- ~5-7 minutes total pipeline
- Videos from existing frames: ~10s

## License

MIT License

## Credits

- Qwen2.5-VL by Alibaba Qwen Team
- Stable Diffusion XL by Stability AI
- FFmpeg for video encoding

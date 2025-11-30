"""
Vision Language Model Components
=================================
FILE: src/vlm/image_analyzer.py
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import Dict, Optional
import json


class ImageAnalyzer:
    """Analyze images using Qwen2.5-VL"""
    
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda"):
        """
        Initialize Image Analyzer with Qwen2.5-VL
        
        Args:
            model_id: HuggingFace model ID
            device: Device to run on
        """
        print(f"Loading Qwen2.5-VL for image analysis: {model_id}")
        
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        print("✓ Image Analyzer initialized")
    
    def analyze_image(self, 
                     image_path: str,
                     analysis_type: str = "detailed") -> Dict[str, str]:
        """
        Analyze an image with different levels of detail
        
        Args:
            image_path: Path to image file
            analysis_type: Type of analysis ("detailed", "brief", "technical")
            
        Returns:
            Dictionary with analysis results
        """
        prompts = {
            "detailed": """Analyze this image in comprehensive detail. Describe:
1. Main subject and its key characteristics
2. Composition, framing, and spatial layout
3. Visual style, color palette, artistic qualities and aesthetic
4. Textures, patterns, and fine details

Be specific, precise, and thorough. Max 1 sentence per step.""",
            
            "brief": """Provide a concise but accurate description of this image, focusing on:
- Main subject
- Key visual elements
- Overall style and mood

Max 1 sentence per step.""",
            
            "technical": """Provide a technical analysis of this image covering:
- Composition structure (rule of thirds, golden ratio, etc.)
- Lighting setup (direction, quality, color temperature)
- Color theory (palette, harmony, contrast)
- Depth and perspective
- Technical quality and rendering style

Max 1 sentence per step."""
        }
        
        prompt = prompts.get(analysis_type, prompts["detailed"])
        print(f"\nAnalyzing image ({analysis_type}): {image_path}")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.open(image_path).convert("RGB")},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        response = self._generate_response(messages)
        
        self.unload()
        
        return {
            "analysis_type": analysis_type,
            "description": response,
            "image_path": image_path
        }
    
    def compare_images(self, 
                      image1_path: str,
                      image2_path: str) -> str:
        """
        Compare two images and describe differences
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            Comparison description
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.open(image1_path).convert("RGB")},
                    {"type": "image", "image": Image.open(image2_path).convert("RGB")},
                    {"type": "text", "text": """Compare these two images and describe:
1. What elements are similar
2. What has changed or is different
3. The progression or transformation between them
4. Visual continuity and consistency

Be precise about specific changes."""}
                ]
            }
        ]
        response = self._generate_response(messages)
        self.unload()
        return response
    
    def extract_key_features(self, image_path: str) -> Dict[str, str]:
        """
        Extract key visual features for prompt engineering
        
        Args:
            image_path: Path to image
            
        Returns:
            Dictionary of extracted features
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.open(image_path).convert("RGB")},
                    {"type": "text", "text": """Extract and list the key visual features:
- Subject: [main subject]
- Style: [artistic style]
- Colors: [dominant colors]
- Lighting: [lighting description]
- Composition: [layout and framing]
- Mood: [overall atmosphere]
- Details: [notable details]

Format each as a single concise phrase."""}
                ]
            }
        ]
        
        response = self._generate_response(messages)
        
        # Parse response into dictionary
        features = {}
        for line in response.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                features[key.strip().lower()] = value.strip()
        
        self.unload()
        
        return features
    
    def unload(self):
        """Move Qwen2.5-VL model to CPU and free GPU memory"""
        self.model.to("cpu")
        torch.cuda.empty_cache()
        print("✓ Qwen2.5-VL moved to CPU")
        
    def move_to_gpu(self):
        """Move Qwen2.5-VL model back to GPU when needed"""
        self.model.to("cuda")
        torch.cuda.empty_cache()
        print("✓ Qwen2.5-VL moved BACK to GPU")
        
    # Extract ONLY the assistant reply, handling all Qwen variations
    def _extract_assistant_text(self, response: str) -> str:
        # Case 1: Qwen chat markers
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant", 1)[1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>", 1)[0]
            return response.strip()

        # Case 2: Model echoes full template without markers
        # Remove everything before the first "assistant"
        if "\nassistant" in response:
            parts = response.split("\nassistant", 1)
            return parts[1].strip()

        # Case 3: Remove system/user echoes
        lines = response.splitlines()
        filtered = []
        skip = True
        for line in lines:
            # Start keeping once model starts speaking naturally
            if line.strip().startswith("Certainly") or line.strip().startswith("The"):
                skip = False
            if not skip:
                filtered.append(line)
        if filtered:
            return "\n".join(filtered).strip()

        # Default fallback
        return response.strip()

    def _generate_response(self, messages: list) -> str:
        """Generate response from VLM"""
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        # # Make sure inputs go to the SAME device as the model
        # model_device = next(self.model.parameters()).device
        # inputs = inputs.to(model_device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=6144,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode
        response = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        response = self._extract_assistant_text(response)        
        return response
    
    

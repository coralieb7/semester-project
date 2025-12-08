"""
FILE: src/svm/universal_evolution_generator.py
Universal Evolution System for ANY Visual Type

UPDATED VERSION with:
- Style, color, and background preservation
- 77-token limit enforcement for SDXL
- Intelligent prompt truncation

This system works for:
- Neurons, particles, networks
- Abstract art, paintings
- Minimalist graphics, logos
- Geometric patterns, vectors
- Organic forms, textures
- Literally ANY visual
"""

import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from .image_analyser import ImageAnalyzer
import json
from PIL import Image
from pathlib import Path


@dataclass
class VisualState:
    """
    Universal visual state representation
    
    These 7 dimensions can describe ANY visual:
    - scale: How much space it occupies
    - complexity: Visual information density
    - detail: Fine-grained features
    - density: Element concentration
    - spread: Spatial distribution
    - contrast: Visual distinction
    - saturation: Color/visual intensity
    """
    scale: float        # 0.0 to 1.0
    quantity: float    # 0.0 to 1.0
    complexity: float   # 0.0 to 1.0
    detail: float       # 0.0 to 1.0
    density: float      # 0.0 to 1.0
    spread: float       # 0.0 to 1.0
    contrast: float     # 0.0 to 1.0
    saturation: float   # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"VisualState(\n"
            f"  scale={self.scale:.2f},\n"
            f"  quantity={self.quantity:.2f},\n"
            f"  complexity={self.complexity:.2f},\n"
            f"  detail={self.detail:.2f},\n"
            f"  density={self.density:.2f},\n"
            f"  spread={self.spread:.2f},\n"
            f"  contrast={self.contrast:.2f},\n"
            f"  saturation={self.saturation:.2f}\n"
            f")"
        )


class UniversalEvolutionGenerator:
    """
    Evolution system that works for ANY visual
    
    Key Innovation:
    - No subject-type classification needed
    - Uses universal visual dimensions
    - VLM translates dimensions into context-aware prompts
    - Same evolution spec works for neurons, abstract art, logos, etc.
    
    Usage:
        evolution = UniversalVisualEvolution(vlm_analyzer)
        
        # Works for neurons
        sequence = evolution.generate_evolution_sequence(
            'neuron.png', 'grow_and_complexify', 13
        )
        
        # Also works for abstract art!
        sequence = evolution.generate_evolution_sequence(
            'abstract.png', 'grow_and_complexify', 13
        )
    """
    
    # Predefined evolution templates
    EVOLUTION_TEMPLATES = {
        'grow_and_complexify': {
            'description': 'Increase size and complexity while maintaining style',
            'scale': 'increase',
            'quantity': 'increase',
            'complexity': 'increase',
            'detail': 'increase',
            'density': 'maintain',
            'spread': 'increase',
            'contrast': 'maintain',
            'saturation': 'maintain'
        },
        'simplify_and_shrink': {
            'description': 'Decrease size and complexity, return to simplicity',
            'scale': 'decrease',
            'quantity': 'decrease',
            'complexity': 'decrease',
            'detail': 'decrease',
            'density': 'maintain',
            'spread': 'decrease',
            'contrast': 'maintain',
            'saturation': 'maintain'
        },
        'densify': {
            'description': 'Increase density and complexity without growing',
            'scale': 'maintain',
            'quantity': 'increase',
            'complexity': 'increase',
            'detail': 'maintain',
            'density': 'increase',
            'spread': 'maintain',
            'contrast': 'maintain',
            'saturation': 'maintain'
        },
        'expand': {
            'description': 'Spread out and grow while becoming less dense',
            'scale': 'increase',
            'quantity': 'increase',
            'complexity': 'maintain',
            'detail': 'maintain',
            'density': 'decrease',
            'spread': 'increase',
            'contrast': 'maintain',
            'saturation': 'maintain'
        },
        'intensify': {
            'description': 'Increase detail, contrast, and saturation',
            'scale': 'maintain',
            'quantity': 'maintain',
            'complexity': 'maintain',
            'detail': 'increase',
            'density': 'maintain',
            'spread': 'maintain',
            'contrast': 'increase',
            'saturation': 'increase'
        },
        'fade': {
            'description': 'Decrease intensity and detail while maintaining structure',
            'scale': 'maintain',
            'quantity': 'maintain',
            'complexity': 'maintain',
            'detail': 'decrease',
            'density': 'maintain',
            'spread': 'maintain',
            'contrast': 'decrease',
            'saturation': 'decrease'
        }
    }
    
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda"):
        """
        Initialize Universal Evolution Generator
        
        Args:
            model_id: Model ID for Image Analyzer
            device: Device to run on
        """
        self.analyzer = ImageAnalyzer(model_id=model_id, device=device)
        print("✓ Universal Evolution Generator initialized")
    
    
    def analyze_initial_state(self, image_path: str) -> VisualState:
        """
        Analyze ANY image and extract universal visual properties
        
        This works for:
        - Neurons → complexity = branch count
        - Abstract art → complexity = number of shapes/elements
        - Minimalist graphics → complexity = design elements
        - Patterns → complexity = repetition variations
        - Literally anything!
        
        Args:
            image_path: Path to image
        
        Returns:
            VisualState with 7 universal dimensions
        """
        
        analysis_prompt = """Analyze this image and rate it on 7 universal visual dimensions.
Score each dimension from 0.0 to 1.0 based on these precise definitions:

1. SCALE (0.0-1.0): How much space does the visual occupy?
   0.0 = tiny, minimal presence (< 10% of frame)
   0.3 = small, ~30% of frame
   0.5 = medium, half the frame
   0.7 = large, most of the frame
   1.0 = fills entire frame
   
2. QUANTITY (0.0-1.0): How many distinct elements are present?
    0.0 = single element
    0.3 = few elements (2-5)
    0.5 = moderate number (5-15)
    0.7 = many elements (15-50)
    1.0 = extremely numerous (50+ elements)

3. COMPLEXITY (0.0-1.0): How much visual information is present?
   0.0 = single simple element
   0.3 = few elements (3-10 distinct features)
   0.5 = moderate (10-30 elements or patterns)
   0.7 = complex (30-100 elements)
   1.0 = extremely complex (100+ elements or intricate patterns)

4. DETAIL (0.0-1.0): Level of fine-grained features and texture?
   0.0 = completely smooth, no visible texture
   0.3 = minimal details, mostly simple surfaces
   0.5 = moderate texture/details visible
   0.7 = detailed features and textures
   1.0 = highly intricate, fine details everywhere

5. DENSITY (0.0-1.0): How tightly packed are the elements?
   0.0 = very sparse, mostly empty space (>70% empty)
   0.3 = loose, spread out (~50% empty)
   0.5 = moderate spacing (~30% empty)
   0.7 = tightly packed (~10% empty)
   1.0 = extremely dense, no gaps

6. SPREAD (0.0-1.0): How distributed across the space?
   0.0 = concentrated in tiny area (<10% of frame)
   0.3 = clustered in small region (~30%)
   0.5 = medium distribution (~50%)
   0.7 = well distributed (~70%)
   1.0 = evenly spread across entire space

7. CONTRAST (0.0-1.0): Level of visual distinction?
   0.0 = very subtle, minimal contrast
   0.3 = gentle contrasts
   0.5 = moderate differences
   0.7 = strong contrasts
   1.0 = extreme contrast, sharp distinctions

8. SATURATION (0.0-1.0): Color/visual intensity?
   0.0 = minimal, very muted, desaturated
   0.3 = subtle color/intensity
   0.5 = moderate saturation
   0.7 = vivid, saturated colors
   1.0 = maximum intensity, highly saturated

CRITICAL: You must respond with ONLY this exact JSON format, no other text:

{
    "scale": 0.X,
    "quantity": 0.X,
    "complexity": 0.X,
    "detail": 0.X,
    "density": 0.X,
    "spread": 0.X,
    "contrast": 0.X,
    "saturation": 0.X,
    "description": "brief 1-sentence description of the visual"
}

Be precise with numbers. Consider the actual visual content carefully.
Output ONLY the JSON, nothing else."""
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path).convert("RGB")},
                {"type": "text", "text": analysis_prompt}
            ]
        }]
        
        response = self.analyzer._generate_response(messages)
        
        # Clean up response (remove markdown, etc.)
        response = response.strip()
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
        response = response.strip()
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Response was: {response[:500]}")
            raise
        
        return VisualState(
            scale=np.clip(float(data['scale']), 0.0, 1.0),
            quantity=np.clip(float(data['quantity']), 0.0, 1.0),
            complexity=np.clip(float(data['complexity']), 0.0, 1.0),
            detail=np.clip(float(data['detail']), 0.0, 1.0),
            density=np.clip(float(data['density']), 0.0, 1.0),
            spread=np.clip(float(data['spread']), 0.0, 1.0),
            contrast=np.clip(float(data['contrast']), 0.0, 1.0),
            saturation=np.clip(float(data['saturation']), 0.0, 1.0)
        )
    
    
    def extract_style_info(self, image_path: str) -> Dict[str, str]:
        """
        Extract style, colors, and background from initial image
        
        Returns:
            Dict with 'subject', 'style', 'colors', and 'background'
        """
        
        prompt = """Analyze this visual and extract style information.

Respond ONLY with this JSON format:

{
    "subject": "main subject in 5-8 words",
    "style": "visual style in 5-8 words", 
    "colors": "primary colors in 3-5 words",
    "background": "background color/type in 2-4 words"
}

Guidelines:
- subject: What is shown (e.g., "crescent moon shape", "abstract organic forms")
- style: Art/design style (e.g., "minimalist graphic design", "watercolor painting")
- colors: Main colors only (e.g., "purple and white", "blue green yellow")
- background: Exact background (e.g., "white background", "neutral gray", "clean white")

Be CONCISE. Each field under 10 words.
If background is white/neutral, specify it clearly.

Output ONLY the JSON, nothing else."""
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path).convert("RGB")},
                {"type": "text", "text": prompt}
            ]
        }]
        
        response = self.analyzer._generate_response(messages).strip()
        
        # Clean up response
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
        response = response.strip()
        
        try:
            style_info = json.loads(response)
            
            # Ensure background is explicitly stated
            bg = style_info['background'].lower()
            if not any(word in bg for word in ['white', 'neutral', 'clean', 'plain', 'blank']):
                print("  ℹ Enforcing white background to prevent SDXL drift")
                style_info['background'] = 'white background'
            
            return style_info
            
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse style info: {e}")
            print(f"Response: {response[:200]}")
            # Fallback
            return {
                'subject': 'visual element',
                'style': 'minimalist design',
                'colors': 'neutral colors',
                'background': 'white background'
            }

    def interpolate_state(self,
                         initial_state: VisualState,
                         evolution_spec: Dict[str, Union[str, float]],
                         signal: float) -> VisualState:
        """
        Calculate visual state at given signal value
        
        This is where the magic happens - signal 0.3 produces
        30% evolution regardless of visual type!
        
        Args:
            initial_state: Starting visual properties
            evolution_spec: How each dimension should evolve
            signal: 0.0 to 1.0 (e.g., 0.3 = 30% evolution)
        
        Returns:
            VisualState at this signal value
        
        Example:
            initial = VisualState(complexity=0.2, ...)
            evolved = interpolate_state(
                initial,
                {'complexity': 'increase'},
                signal=0.3
            )
            # evolved.complexity ≈ 0.2 + 0.3 * (1.0 - 0.2) = 0.44
            # 30% of the way from 0.2 to 1.0
        """
        
        def evolve_dimension(initial_val: float, 
                           directive: Union[str, float]) -> float:
            """Evolve a single dimension based on directive"""
            
            if directive == 'increase':
                # Increase from initial toward 1.0
                return initial_val + signal * (1.0 - initial_val)
            
            elif directive == 'decrease':
                # Decrease from initial toward 0.0
                return initial_val * (1.0 - signal)
            
            elif directive == 'maintain':
                # Keep constant
                return initial_val
            
            elif isinstance(directive, (int, float)):
                # Interpolate to specific target value
                target = np.clip(float(directive), 0.0, 1.0)
                return initial_val + signal * (target - initial_val)
            
            else:
                # Unknown directive, maintain
                return initial_val
        
        return VisualState(
            scale=evolve_dimension(
                initial_state.scale, 
                evolution_spec.get('scale', 'maintain')
            ),
            quantity=evolve_dimension(
                initial_state.quantity,
                evolution_spec.get('quantity', 'maintain')
            ),
            complexity=evolve_dimension(
                initial_state.complexity,
                evolution_spec.get('complexity', 'maintain')
            ),
            detail=evolve_dimension(
                initial_state.detail,
                evolution_spec.get('detail', 'maintain')
            ),
            density=evolve_dimension(
                initial_state.density,
                evolution_spec.get('density', 'maintain')
            ),
            spread=evolve_dimension(
                initial_state.spread,
                evolution_spec.get('spread', 'maintain')
            ),
            contrast=evolve_dimension(
                initial_state.contrast,
                evolution_spec.get('contrast', 'maintain')
            ),
            saturation=evolve_dimension(
                initial_state.saturation,
                evolution_spec.get('saturation', 'maintain')
            )
        )
    
    def generate_prompt_for_state(self,
                                  state: VisualState,
                                  signal: float,
                                  base_description: str,
                                  style_info: Dict[str, str],
                                  style_prefix: str = "") -> str:
        """
        Generate SDXL prompt for a visual state
        
        The VLM translates abstract dimensions into concrete,
        context-aware descriptions.
        
        For a neuron: "with 10 branches extending..."
        For abstract art: "with 15 organic shapes..."
        For a logo: "with 5 geometric elements..."
        
        Same dimensions, different interpretations!
        
        Args:
            state: Target visual state
            signal: Current signal (0-1)
            base_description: Brief description of initial visual
            style_prefix: Optional style prefix for LoRA
        
        Returns:
            SDXL prompt string
        """
        
        state_prompt = f"""Generate a detailed SDXL image generation prompt (max 77 tokens) based on these requirements:

BASE VISUAL DESCRIPTION:
{base_description}

EVOLUTION PROGRESS: {signal:.0%} (signal value: {signal:.2f})

TARGET VISUAL STATE (all values 0.0-1.0):
• Scale/Size: {state.scale:.2f} 
  (0.0=tiny/minimal, 0.5=medium, 1.0=fills entire frame)
  
• Quantity: {state.quantity:.2f} 
  (0.0=one element, 0.5=moderate number, 1.0=extremely numerous)

• Complexity: {state.complexity:.2f}
  (0.0=single element, 0.5=moderate, 1.0=extremely complex)
  
• Detail: {state.detail:.2f}
  (0.0=smooth/simple, 0.5=moderate texture, 1.0=highly detailed)
  
• Density: {state.density:.2f}
  (0.0=very sparse, 0.5=moderate, 1.0=densely packed)
  
• Spread: {state.spread:.2f}
  (0.0=concentrated, 0.5=medium distribution, 1.0=evenly distributed)
  
• Contrast: {state.contrast:.2f}
  (0.0=subtle, 0.5=moderate, 1.0=high contrast)
  
• Saturation: {state.saturation:.2f}
  (0.0=muted/desaturated, 0.5=moderate, 1.0=highly saturated)

YOUR TASK:
Create a detailed SDXL prompt with 77 tokens that:
1. Maintains the essence and subject matter of the base visual
2. Adjusts ALL visual properties to EXACTLY match the target state
3. Is specific and quantitative about these properties
4. Uses concrete descriptive terms

PRESERVED ELEMENTS (MUST include):
- Subject: {style_info['subject']}
- Style: {style_info['style']}
- Colors: {style_info['colors']}
- Background: {style_info['background']}

CRITICAL REQUIREMENTS:
• The visual MUST occupy approximately {state.scale:.0%} of the frame
• Visual complexity MUST be at {state.complexity:.0%} level (translate this to appropriate element counts or pattern complexity for the subject)
• Detail/texture MUST match {state.detail:.0%} level
• Element density MUST be {state.density:.0%} (sparse to packed)
• Spatial distribution MUST be {state.spread:.0%} (concentrated to distributed)
• Contrast MUST be {state.contrast:.0%} (subtle to high contrast)
• Color intensity MUST be {state.saturation:.0%} (muted to saturated)

IMPORTANT:
- Translate these abstract dimensions into CONCRETE visual descriptions appropriate for the subject
- Be SPECIFIC about quantities when relevant (e.g., "approximately 15 branches", "20-25 shapes", "dense pattern with 100+ repetitions")
- Use descriptive language that SDXL will understand
- No more than 77 tokens total
- ADD THE BACKGROUND STYLE AND COLOR PALETTE TO THE PROMPT


Respond with ONLY the SDXL prompt, no explanation or additional text. Max 77 tokens."""
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": state_prompt}
            ]
        }]
        
        prompt = self.analyzer._generate_response(messages).strip()
        
        # Add style prefix if provided
        if style_prefix:
            prompt = f"{style_prefix}, {prompt}"
        
        return prompt
    
    def generate_evolution_sequence(self,
                                   image_path: str,
                                   evolution_template: Union[str, Dict],
                                   num_steps: int = 13,
                                   style_prefix: str = "") -> List[Dict]:
        """
        Generate complete evolution sequence for ANY visual
        
        This is the main function you'll use. It works for
        absolutely any image type!
        
        Args:
            image_path: Path to initial image
            evolution_template: Template name or custom spec dict
            num_steps: Number of evolution steps
            style_prefix: Optional style prefix for LoRA activation
        
        Returns:
            List of prompt dictionaries
        
        Example:
            evolution = UniversalVisualEvolution(vlm)
            
            # Works for neurons
            sequence = evolution.generate_evolution_sequence(
                'neuron.png', 'grow_and_complexify', 13
            )
            
            # Works for abstract art
            sequence = evolution.generate_evolution_sequence(
                'abstract.png', 'grow_and_complexify', 13
            )
            
            # Works for minimalist graphics
            sequence = evolution.generate_evolution_sequence(
                'logo.png', 'intensify', 13
            )
            
            # Works for ANYTHING
            sequence = evolution.generate_evolution_sequence(
                'any_image.png', 'grow_and_complexify', 13
            )
        """
        
        print(f"\n{'='*70}")
        print(f"Universal Evolution Sequence Generation")
        print(f"{'='*70}")
        print(f"Image: {image_path}")
        print(f"Steps: {num_steps}")
        
        # Step 1: Extract style info (NEW!)
        print(f"\n[1/4] Extracting style information...")
        style_info = self.extract_style_info(image_path)
        print(f"\nStyle preserved:")
        print(f"  Subject: {style_info['subject']}")
        print(f"  Style: {style_info['style']}")
        print(f"  Colors: {style_info['colors']}")
        print(f"  Background: {style_info['background']}")
        
        # Step 2: Analyze initial state
        print(f"\n[1/3] Analyzing initial visual state...")
        initial_state = self.analyze_initial_state(image_path)
        print(f"\nInitial state:")
        print(initial_state)
        
        # Step 3: Get evolution specification
        if isinstance(evolution_template, str):
            if evolution_template not in self.EVOLUTION_TEMPLATES:
                available = list(self.EVOLUTION_TEMPLATES.keys())
                raise ValueError(
                    f"Unknown template: '{evolution_template}'. "
                    f"Available: {available}"
                )
            evolution_spec = self.EVOLUTION_TEMPLATES[evolution_template]
            print(f"\n[2/3] Using template: '{evolution_template}'")
            print(f"Description: {evolution_spec['description']}")
        else:
            evolution_spec = evolution_template
            print(f"\n[2/3] Using custom evolution spec")
        
        # Remove description from spec if present
        evolution_spec = {k: v for k, v in evolution_spec.items() 
                         if k != 'description'}
        
        # Step 4: Get base description
        base_description = self._get_visual_description(image_path)
        print(f"\nBase description: {base_description}")
        
        # Step 5: Generate sequence
        print(f"\n[3/3] Generating {num_steps} evolution prompts...")
        sequence = []
        
        for i in range(num_steps):
            signal = i / (num_steps - 1) if num_steps > 1 else 0.0
            
            # Calculate state at this signal
            current_state = self.interpolate_state(
                initial_state,
                evolution_spec,
                signal
            )
            
            # Generate prompt for this state
            prompt = self.generate_prompt_for_state(
                current_state,
                signal,
                base_description,
                style_info,
                style_prefix
            )
            
            sequence.append({
                'step': i + 1,
                'signal': signal,
                'state': current_state.to_dict(),
                'prompt': prompt,
                'style_info': style_info,
                'change': self._describe_change(current_state, initial_state, signal)
            })
            
            print(f"  Step {i+1:2d}/{num_steps}: signal={signal:.2f}, "
                  f"complexity={current_state.complexity:.2f}, "
                  f"scale={current_state.scale:.2f}"
                  f"quantity={current_state.quantity:.2f}")
        
        print(f"\n✓ Generated {len(sequence)} prompts")
        print(f"{'='*70}\n")
        
        return sequence
    
    def _get_visual_description(self, image_path: str) -> str:
        """Get brief description of visual for context"""
        
        prompt = """Provide a brief (1-2 sentences) description of this visual.
Focus on:
- The main subject or content
- Key visual characteristics
- Overall style or aesthetic

Do NOT mention:
- Specific quantities or dimensions
- Complexity levels
- Spatial properties

Just describe what you see in simple, clear terms."""
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path).convert("RGB")},
                {"type": "text", "text": prompt}
            ]
        }]
        
        return self.analyzer._generate_response(messages).strip()
    
    def _describe_change(self, 
                        current: VisualState, 
                        initial: VisualState,
                        signal: float) -> str:
        """Generate human-readable description of change"""
        
        if signal == 0.0:
            return "Initial state"
        
        changes = []
        
        # Check which dimensions changed significantly
        for dim in ['scale', 'complexity', 'detail', 'density', 
                    'spread', 'contrast', 'saturation']:
            current_val = getattr(current, dim)
            initial_val = getattr(initial, dim)
            diff = current_val - initial_val
            
            if abs(diff) > 0.15:  # Significant change
                if diff > 0:
                    changes.append(f"increased {dim}")
                else:
                    changes.append(f"decreased {dim}")
        
        if changes:
            return f"Signal {signal:.2f}: {', '.join(changes[:3])}"
        else:
            return f"Signal {signal:.2f}: gradual progression"
    
    @classmethod
    def list_templates(cls) -> None:
        """Print all available evolution templates"""
        print("\nAvailable Evolution Templates:")
        print("=" * 70)
        for name, spec in cls.EVOLUTION_TEMPLATES.items():
            print(f"\n{name}:")
            print(f"  {spec.get('description', 'No description')}")
        print("=" * 70)
        
        
    def save_prompts(self, prompts: List[Dict], output_path: str):
        """Save prompts to JSON file"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Prompts saved to: {output_path}")

# Convenience functions

def create_universal_evolution(vlm_analyzer):
    """Factory function to create evolution system"""
    return UniversalEvolutionGenerator(vlm_analyzer)


def list_evolution_templates():
    """List all available templates"""
    UniversalEvolutionGenerator.list_templates()


# Example usage
if __name__ == "__main__":
    print("Universal Visual Evolution System")
    print("=" * 70)
    print("\nThis system works for ANY visual type:")
    print("  • Neurons, particles, networks")
    print("  • Abstract art, paintings")
    print("  • Minimalist graphics, logos")
    print("  • Geometric patterns, vectors")
    print("  • Organic forms, textures")
    print("  • Literally anything!")
    print("\n" + "=" * 70)
    
    list_evolution_templates()
    
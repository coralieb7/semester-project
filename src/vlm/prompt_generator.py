"""
FILE: src/vlm/prompt_generator.py
"""

import json, re
from typing import List, Dict, Optional
from pathlib import Path
from .image_analyser import ImageAnalyzer


class PromptGenerator:
    """Generate evolutionary prompts using VLM"""
    
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda"):
        """
        Initialize Prompt Generator
        
        Args:
            model_id: Model ID for VLM
            device: Device to run on
        """
        self.analyzer = ImageAnalyzer(model_id=model_id, device=device)
        print("✓ Prompt Generator initialized")
    
    def expand_evolution_goal(self,
                             evolution_goal: str,
                             style_guidelines: Optional[Dict[str, str]] = None) -> str:
        """
        Expand evolution goal with detailed descriptions and style guidelines
        
        Args:
            evolution_goal: Brief evolution goal (e.g., "a beautiful sunflower")
            style_guidelines: Optional dict with style rules
            
        Returns:
            Detailed expanded description
        """
        print(f"\n{'='*70}")
        print("Expanding Evolution Goal with Style Guidelines")
        print(f"{'='*70}")
        print(f"Original goal: {evolution_goal}")
        
        # Default style guidelines
        if style_guidelines is None:
            style_guidelines = {
                "background": "white background",
                "composition": "centered, balanced composition",
                "color_palette": "vibrant, harmonious colors",
                "style": "clean, modern aesthetic",
                "details": "high detail, professional quality"
            }
        
        # Build style context
        style_context = "\n".join([f"- {k}: {v}" for k, v in style_guidelines.items()])
        
        expansion_prompt = f"""You are an expert visual designer and art director.

TASK: Expand and enrich this brief concept into a detailed, vivid visual description.

BRIEF CONCEPT:
"{evolution_goal}"

STYLE GUIDELINES TO FOLLOW:
{style_context}

YOUR JOB:
1. Elaborate on the concept with rich visual details
2. Specify colors, textures, forms, and spatial arrangement
3. Add atmospheric qualities (lighting, mood)
4. Incorporate ALL the style guidelines naturally
5. Make it specific enough for an AI image generator

REQUIREMENTS:
- Output 2-4 sentences maximum
- Be concrete and visual (not abstract or poetic)
- Include specific colors, shapes, and composition details
- Naturally integrate the style guidelines
- Make it sound like a complete SDXL prompt

EXAMPLES:
Input: "a beautiful sunflower"
Output: "A large, vibrant sunflower with golden yellow petals radiating from a dark brown center, positioned centrally on a pure white background. The flower head fills 60% of the frame, with soft natural lighting highlighting the delicate petal textures and seed patterns. Clean, minimalist composition with high botanical detail and warm, inviting color palette."

Input: "abstract flowing shapes"
Output: "Smooth, organic purple shapes flowing diagonally across the composition, with soft gradients and translucent overlays creating depth. Centered on a clean white background with subtle shadows for dimensionality. Modern, minimalist aesthetic with fluid forms and harmonious color transitions from deep violet to soft lavender."

Now expand this:
"{evolution_goal}"

Output ONLY the expanded description, no other text."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": expansion_prompt}
                ]
            }
        ]
        
        # Ensure VLM is loaded
        self.analyzer.move_to_gpu()
        
        response = self.analyzer._generate_response(messages)
        
        # Clean up response
        expanded_description = response.strip()
        
        # Remove any quotes or preamble
        if expanded_description.startswith('"') and expanded_description.endswith('"'):
            expanded_description = expanded_description[1:-1]
        
        print(f"\nExpanded description:")
        print(f"  {expanded_description}")
        print(f"✓ Expansion complete ({len(expanded_description)} chars)")
        
        return expanded_description
    
    def generate_evolutionary_sequence(self,
                                      image_path: str,
                                      evolution_description: str,
                                      num_prompts: int = 13,
                                      visual_guidelines: Optional[str] = None,
                                      analysis_type: str = "brief",
                                      prompts_type: str = 'basic',
                                      style_guidelines: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Generate evolutionary prompt sequence for reverse engineering approach
        
        Args:
            image_path: Path to initial/target image
            evolution_description: Description of desired evolution
            num_prompts: Number of prompts to generate (max sequence length)
            visual_guidelines: Optional visual guidelines to follow
            
        Returns:
            List of prompt dictionaries with step, prompt, and metadata
        """
        print(f"\n{'='*70}")
        print(f"Generating {num_prompts} evolutionary prompts")
        print(f"{'='*70}")
        
        # Step 1: Analyze the image
        print("\n[1/2] Analyzing initial image...")
        analysis = self.analyzer.analyze_image(image_path, analysis_type=analysis_type)
        image_description = analysis['description']
        print(image_description)
        print(f"Image analysis complete ({len(image_description)} chars)")

        
        #Step 2: Expand evolution description
        print("\nExpanding evolution description...")
        evolution_description = self.expand_evolution_goal(
            evolution_description,
            style_guidelines=style_guidelines
        )
        
        # Step 3: Generate evolutionary sequence
        print(f"\n[2/2] Generating {num_prompts} evolutionary prompts...")
        
        visual_section = ""
        if visual_guidelines:
            visual_section = f"VISUAL TO FOLLOW:\n{visual_guidelines}"


        evolution_prompt_basic = f"""You are an expert at creating evolutionary image generation prompts.

                                IMAGE ANALYSIS (STATE A):
                                {image_description}

                                EVOLUTION GOAL (STATE B):
                                {evolution_description}

                                Your job is to design a smooth visual morph from STATE A to STATE B over EXACTLY {num_prompts} steps.

                                INTERPRETATION OF THE TASK:
                                - Think of the sequence as a gradual transformation from the analyzed image into the evolution goal: {evolution_description}
                                - Step 1 should look almost exactly like STATE A and barely like STATE B.
                                - The last step (step {num_prompts}) should look almost exactly like STATE B ({evolution_description}) and barely like STATE A.
                                - Intermediate steps should be mixtures: early steps are closer to STATE A, later steps closer to STATE B ({evolution_description}).

                                EVOLUTION BEHAVIOR (MANDATORY):
                                For each step i in 1..{num_prompts}:
                                - Visually, the scene should become LESS like STATE A and MORE like STATE B.
                                - Each step must introduce at least ONE new visual element, attribute, or structural change that clearly moves it toward STATE B.
                                - Do NOT just tweak wording or adjectives; the underlying visual content must evolve.
                                - Avoid repeating the same description across many steps.

                                CRITICAL REQUIREMENTS:
                                1. Generate EXACTLY {num_prompts} steps (no more, no less).
                                2. Each "prompt" must be a complete, standalone SDXL prompt.
                                3. The first prompt (step 1) should closely match the analyzed image (STATE A).
                                4. The last prompt (step {num_prompts}) should closely match the evolution goal: {evolution_description} (STATE B).
                                5. Changes between consecutive steps should be SUBTLE but REAL and CUMULATIVE.
                                6. Each prompt should include: subject, style, colors, lighting, and composition details.
                                7. Across the sequence, the subject, structure, or arrangement should clearly transform in a way that reflects the evolution goal.
                                8. The "change" field for each step must describe how this step moves visually closer to STATE B compared to the previous step.

                                IMPORTANT STYLE CONSTRAINTS:
                                - Keep prompts concise (max 100 tokens per step).
                                - The "change" field must be a single short sentence focused on the transformation from the previous step.
                                - Do NOT rewrite the entire image analysis in each prompt.
                                - Avoid staying stuck on the original description; the influence of STATE A must clearly fade over time, while elements related to STATE B become more prominent.
                                
                                STYLISTIC GUIDELINES:
                                - The background must be WHITE
                                - The visual is CENTERED
                                - The color is PURPLE
                                - NO ARTIFACT
                                
                                FORMAT:
                                Output ONLY a JSON array with EXACTLY this structure:
                                [
                                {{
                                    "step": 1,
                                    "prompt": "complete detailed SDXL prompt for step 1 (very close to STATE A)",
                                    "change": "how this is different from STATE A but still very close",
                                    "progress": "0.0"
                                }},
                                {{
                                    "step": 2,
                                    "prompt": "complete detailed SDXL prompt for step 2 (slightly closer to STATE B)",
                                    "change": "what changed to move closer to STATE B",
                                    "progress": "0.05"
                                }},
                                ...
                                {{
                                    "step": {num_prompts-2},
                                    "prompt": "complete detailed SDXL prompt close to STATE B",
                                    "change": "final adjustments that fully realize STATE B",
                                    "progress": "1.0"
                                }},
                                {{
                                    "step": {num_prompts-1},
                                    "prompt": "complete detailed SDXL prompt almost at STATE B",
                                    "change": "final adjustments that fully realize STATE B",
                                    "progress": "1.0"
                                }},
                                {{
                                    "step": {num_prompts},
                                    "prompt": "complete detailed SDXL prompt matching the evolution goal STATE B",
                                    "change": "final adjustments that fully realize STATE B",
                                    "progress": "1.0"
                                }}
                                ]

                                The "progress" field should go from 0.0 to 1.0 approximately linearly.
                                Output ONLY the JSON array, with no extra text before or after.
                                """
                                
        evolution_prompt_only = f"""You are an expert at creating evolutionary image generation prompts.

                                    IMAGE ANALYSIS (for style reference only):
                                    {image_description}

                                    EVOLUTION GOAL (primary focus):
                                    {evolution_description}

                                    YOUR TASK:
                                    Design a progressive sequence that BUILDS toward the evolution goal: "{evolution_description}"
                                    The image analysis provides STYLE REFERENCE ONLY (colors, composition, aesthetic).

                                    INTERPRETATION:
                                    - This is NOT a morphing from image to goal
                                    - This is a PROGRESSION that develops the evolution goal over {num_prompts} steps
                                    - Extract STYLE ELEMENTS from the image (colors, composition, aesthetic approach)
                                    - Apply those style elements to the evolution goal at each step
                                    - Each step should show the evolution goal becoming more developed, detailed, and refined

                                    STYLE EXTRACTION FROM IMAGE:
                                    Analyze the image and extract:
                                    1. Color palette and color treatment
                                    2. Composition style (centered, balanced, minimal, etc.)
                                    3. Background approach (solid color, gradient, textured, etc.)
                                    4. Visual aesthetic (flat, 3D, illustrative, photorealistic, etc.)
                                    5. Level of detail and rendering style

                                    EVOLUTION STRATEGY:
                                    - Step 1: Simple, minimal version of "{evolution_description}" using extracted style
                                    - Steps 2-{num_prompts//3}: Gradually add main structural elements
                                    - Steps {num_prompts//3+1}-{2*num_prompts//3}: Refine details and complexity
                                    - Steps {2*num_prompts//3+1}-{num_prompts}: Polish and perfect the final vision

                                    EVOLUTION BEHAVIOR (MANDATORY):
                                    For each step i in 1..{num_prompts}:
                                    - The subject is ALWAYS about "{evolution_description}"
                                    - Apply the STYLE from the analyzed image (colors, composition, aesthetic)
                                    - Each step adds MORE detail, complexity, or refinement to the evolution goal
                                    - Do NOT reference the original image's subject matter
                                    - Focus on BUILDING UP the evolution goal progressively

                                    EXAMPLE (to clarify the approach):
                                    Image analysis: "A purple circle on white background, minimalist flat design"
                                    Evolution goal: "a beautiful sunflower"

                                    Step 1: "A simple purple circular flower center on white background, minimalist flat design"
                                    Step 2: "Purple circular flower center with first layer of petals beginning to form, centered on white, flat style"
                                    Step 3: "Developing purple sunflower with partial petal ring, clean modern aesthetic, white background"
                                    ...
                                    Step {num_prompts}: "Fully detailed purple sunflower with complete petal formation, seeds visible in center, minimalist flat graphic design, centered on white background"

                                    CRITICAL REQUIREMENTS:
                                    1. Generate EXACTLY {num_prompts} steps (no more, no less).
                                    2. Each "prompt" must be a complete, standalone SDXL prompt.
                                    3. ALL steps should be about the evolution goal, NOT the original image subject.
                                    4. Apply the STYLE from the image (colors, composition, aesthetic) to the evolution goal.
                                    5. Changes should show DEVELOPMENT and REFINEMENT of the evolution goal.
                                    6. Each step should be MORE complete than the previous one.
                                    7. The final step should be the fully realized evolution goal in the extracted style.

                                    STYLE CONSTRAINTS TO EXTRACT AND APPLY:
                                    - Use the color palette from the analyzed image
                                    - Use the composition approach (centered, balanced, etc.)
                                    - Use the background style (white, solid color, etc.)
                                    - Use the aesthetic approach (minimalist, detailed, etc.)

                                    IMPORTANT:
                                    - Keep prompts concise (max 100 tokens per step).
                                    - The "change" field must describe what was added or refined in this step.
                                    - Focus on PROGRESSIVE DEVELOPMENT, not morphing or transformation.
                                    - The subject is the evolution goal from step 1 onwards.

                                    FORMAT:
                                    Output ONLY a JSON array with EXACTLY this structure:
                                    [
                                    {{
                                        "step": 1,
                                        "prompt": "simple initial version of evolution goal in extracted style",
                                        "change": "initial minimal form",
                                        "progress": "0.0"
                                    }},
                                    {{
                                        "step": 2,
                                        "prompt": "evolution goal with first additions in extracted style",
                                        "change": "what structural element was added",
                                        "progress": "{1.0/(num_prompts-1):.3f}"
                                    }},
                                    ...
                                    {{
                                        "step": {num_prompts-1},
                                        "prompt": "nearly complete evolution goal in extracted style",
                                        "change": "final refinements being added",
                                        "progress": "{(num_prompts-2)/(num_prompts-1):.3f}"
                                    }},
                                    {{
                                        "step": {num_prompts},
                                        "prompt": "fully realized evolution goal matching extracted style perfectly",
                                        "change": "final polished state achieved",
                                        "progress": "1.0"
                                    }}
                                    ]

                                    The "progress" field should go from 0.0 to 1.0 linearly.
                                    Output ONLY the JSON array, with no extra text before or after.
                                    """

        evolution_prompt_few_shot = f"""You are an expert at imagining how visual scenes can naturally evolve over time.

                                    You will be given an IMAGE DESCRIPTION.  
                                    Your task is to imagine what the scene *could become* if it progressively transforms in a coherent way.

                                    Your mission is to generate a sequence of EXACTLY {num_prompts} prompts that gradually evolves the scene from its initial form into a more advanced, expanded, or exaggerated version that still makes sense visually.

                                    ------------------------------------------------------------
                                    FEW-SHOT CONTEXT (LEARN THESE PATTERNS)
                                    ------------------------------------------------------------                                    
                                    EXAMPLE 1:
                                    Initial description:
                                    "A small pile of rocks in a barren desert."

                                    Reasonable evolution:
                                    - Step 1: small pile
                                    - Step 2: more rocks accumulate
                                    - Step 3: pile becomes a mound
                                    - Step 4: mound grows into a hill
                                    - Step 5: hill becomes a massive rocky mountain
                                    
                                    EXAMPLE 2:
                                    Initial description:
                                    "A paint stain with small droplets scattered around it."

                                    Reasonable evolution:
                                    - Step 1: small stain
                                    - Step 2: droplets merge into the stain
                                    - Step 3: stain becomes a large puddle
                                    ...
                                    - Step {num_prompts}: puddle overflows and spreads across the surface

                                    EXAMPLE 3:
                                    Initial description:
                                    "A tiny glowing spark floating in darkness."

                                    Reasonable evolution:
                                    - Step 1: faint spark
                                    - Step 2: spark becomes brighter
                                    - Step 3: glow expands outward
                                    - Step 4: spark becomes a flame
                                    - Step 5: flame becomes huge
                                    - Step 6: flame grows into a swirling vortex of fire

                                    ------------------------------------------------------------
                                    PATTERN YOU MUST LEARN:
                                    - Start close to the original scene.
                                    - Each step adds one plausible visual change.
                                    - Evolution must feel natural: growth, expansion, intensification, transformation.
                                    - Last step is a more developed, dramatic, or expanded form of the initial subject.
                                    ------------------------------------------------------------

                                    NOW APPLY THIS LOGIC TO THE ACTUAL IMAGE:

                                    IMAGE DESCRIPTION:
                                    {image_description}

                                    EVOLUTION GOAL (high-level idea of direction):
                                    {evolution_description}

                                    TASK:
                                    Generate EXACTLY {num_prompts} evolutionary prompts that:
                                    - begin by closely matching the analyzed image,
                                    - evolve toward the evolution goal,
                                    - follow the same transformation logic as the examples.
                                    
                                    STYLISTIC GUIDELINES:
                                    - The background must be WHITE
                                    - The visual is CENTERED
                                    - The color is PURPLE
                                    - NO ARTIFACT

                                    FORMAT:
                                    Output ONLY a JSON array with EXACTLY this structure:
                                    [
                                    {{
                                        "step": 1,
                                        "prompt": "a detailed prompt matching the original scene",
                                        "change": "initial state",
                                        "progress": "0.0"
                                    }},
                                    {{
                                        "step": 2,
                                        "prompt": " a SDXL prompt with a slight evolution",
                                        "change": "small meaningful change",
                                        "progress": "0.05"
                                    }},
                                    ...
                                    {{
                                        "step": {num_prompts},
                                        "prompt": "a complete prompt fully achieving the evolution",
                                        "change": "final transformation",
                                        "progress": "1.0"
                                    }}
                                    ]

                                    The "progress" field should go from 0.0 to 1.0 approximately linearly.
                                    Output ONLY the JSON array, with no extra text before or after.

                                    """
        evolution_prompt_matrix = f"""
                                    You are designing an evolutionary sequence of SDXL prompts.

                                    The image can be thought of as STATE A.  
                                    The evolution goal is STATE B.  

                                    EVOLUTION PRINCIPLE (VERY IMPORTANT):
                                    Each step must be a MIX of A and B, where the weight of A decreases and the weight of B increases.

                                    Define:
                                    - A_WEIGHT(i) = how much the step resembles the original image
                                    - B_WEIGHT(i) = how much the step resembles the evolution goal

                                    They must satisfy:
                                    A_WEIGHT(1) = 1.0
                                    B_WEIGHT(1) = 0.0

                                    A_WEIGHT({num_prompts}) = 0.0
                                    B_WEIGHT({num_prompts}) = 1.0

                                    For each intermediate step:
                                    A_WEIGHT(i) decreases linearly.
                                    B_WEIGHT(i) increases linearly.

                                    You MUST explicitly incorporate these weights in the prompt design:
                                    - Prompts MUST change in structure, geometry, and visual composition.
                                    - Prompts MUST NOT start with similar sentences or phrasing.
                                    - Prompts MUST NOT repeat parts of previous steps.

                                    IMAGE DESCRIPTION (STATE A):
                                    {image_description}

                                    EVOLUTION GOAL (STATE B):
                                    {evolution_description}

                                    TASK:
                                    Generate EXACTLY {num_prompts} prompts.  
                                    For each step i:
                                    - Describe the image as a blend of A and B according to the weights.
                                    - Add ONE major structural or geometric change per step.
                                    - Ensure each prompt is visually DIFFERENT from the previous one.
                                    - Ensure each new step adds NEW elements from B.
                                    - Keep prompts concise (max 100 tokens).

                                    FORMAT:
                                    [
                                    {{
                                        "step": 1,
                                        "prompt": "SDXL prompt using the weighted mixture",
                                        "change": "initial state",
                                        "progress": "0.0"
                                    }},
                                    ...
                                    {{
                                        "step": i,
                                        "prompt": "SDXL prompt using the weighted mixture",
                                        "change": "what changed structurally from step i-1",
                                        "progress": "<0.0–1.0>"
                                    }},
                                    ...
                                    {{
                                        "step": {num_prompts},
                                        "prompt": "SDXL prompt using the weighted mixture",
                                        "change": "final state",
                                        "progress": "1.0"
                                    }}
                                    ]

                                    FORMAT:
                                    Output ONLY a JSON array with EXACTLY this structure:
                                    """
                                    
        if prompts_type == 'basic':
            evolution_prompt = evolution_prompt_basic
        elif prompts_type == 'few_shot':
            evolution_prompt = evolution_prompt_few_shot
        elif prompts_type == 'matrix':
            evolution_prompt = evolution_prompt_matrix
        elif prompts_type == 'only':
            evolution_prompt = evolution_prompt_only
            
            
        print(f"Generating prompts with strategy {prompts_type}")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": evolution_prompt}
                ]
            }
        ]
        # Move Qwen back to GPU 
        self.analyzer.move_to_gpu()
        
        response = self.analyzer._generate_response(messages)
        
        # Parse JSON response
        try:
            prompts = self._parse_json_response(response, num_prompts)
            
            # Validate we got the right number
            if len(prompts) != num_prompts:
                print(f"⚠ Warning: Got {len(prompts)} prompts instead of {num_prompts}")
                prompts = self._adjust_prompt_count(prompts, num_prompts)
            
            # Ensure progress values are correct
            for i, prompt in enumerate(prompts):
                prompt['progress'] = float(i) / float((len(prompts) - 1)) if len(prompts) > 1 else 0.0
            
            print(f"✓ Generated {len(prompts)} prompts successfully")
            self.analyzer.unload()
            return prompts
            
        except Exception as e:
            print(f"✗ Error parsing JSON response: {e}")
            print(f"Raw response length: {len(response)} chars")
            print(response[:1000])
            
            # Fallback: create simple evolutionary prompts
            print("Using fallback prompt generation...")
            self.analyzer.unload()
            return self._create_fallback_prompts(
                image_description,
                evolution_description,
                num_prompts
            )
    
    def _parse_json_response(self, response: str, expected_count: int) -> List[Dict]:
        """Parse JSON from VLM response"""
        
        # Try to extract JSON array
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON array found in response")
        
        json_str = response[start_idx:end_idx]
        
        # Simple cleanup: remove trailing commas before ] or }
        json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
    
        prompts = json.loads(json_str)
        
        # Validate structure
        if not isinstance(prompts, list):
            raise ValueError("Response is not a list")
        
        for prompt in prompts:
            if not all(key in prompt for key in ['step', 'prompt', 'change']):
                raise ValueError("Missing required fields in prompt")
        
        return prompts
    
    def _adjust_prompt_count(self, prompts: List[Dict], target_count: int) -> List[Dict]:
        """Adjust prompt list to match target count"""
        
        if len(prompts) > target_count:
            # Remove middle prompts to reach target
            step = len(prompts) / target_count
            indices = [int(i * step) for i in range(target_count)]
            prompts = [prompts[i] for i in indices]
        
        elif len(prompts) < target_count:
            # Duplicate and interpolate
            while len(prompts) < target_count:
                # Insert between existing prompts
                new_prompts = []
                for i in range(len(prompts) - 1):
                    new_prompts.append(prompts[i])
                    if len(new_prompts) + len(prompts) - i - 1 < target_count:
                        # Create interpolated prompt
                        interp = {
                            'step': len(new_prompts) + 1,
                            'prompt': f"{prompts[i]['prompt']}, gradually transitioning",
                            'change': f"Intermediate step between {prompts[i]['step']} and {prompts[i+1]['step']}",
                            'progress': (prompts[i].get('progress', 0) + prompts[i+1].get('progress', 1)) / 2
                        }
                        new_prompts.append(interp)
                new_prompts.append(prompts[-1])
                prompts = new_prompts
        
        # Renumber steps
        for i, prompt in enumerate(prompts, 1):
            prompt['step'] = i
        
        return prompts
    
    def _create_fallback_prompts(self,
                                image_description: str,
                                evolution_description: str,
                                num_prompts: int) -> List[Dict]:
        """Create simple fallback prompts if JSON parsing fails"""
        
        # Extract key terms from descriptions
        base_terms = image_description[:200]
        
        prompts = []
        for i in range(num_prompts):
            progress = i / (num_prompts - 1) if num_prompts > 1 else 0.0
            intensity = int(progress * 100)
            
            prompts.append({
                'step': i + 1,
                'prompt': f"{base_terms}, {evolution_description}, progression {intensity}%, detailed, high quality",
                'change': f"Step {i+1}: {intensity}% evolved towards target",
                'progress': progress
            })
        
        return prompts
    
    def save_prompts(self, prompts: List[Dict], output_path: str):
        """Save prompts to JSON file"""
        
        output_path = Path("data/output/prompts") / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Prompts saved to: {output_path}")
    
    def load_prompts(self, input_path: str) -> List[Dict]:
        """Load prompts from JSON file"""
        
        with open(input_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        
        print(f"✓ Loaded {len(prompts)} prompts from: {input_path}")
        return prompts
    
    def display_prompts(self, prompts: List[Dict]):
        """Display prompts in readable format"""
        
        print(f"\n{'='*70}")
        print(f"EVOLUTIONARY PROMPT SEQUENCE ({len(prompts)} steps)")
        print(f"{'='*70}")
        
        for prompt_data in prompts:
            print(f"\n[Step {prompt_data['step']}] Progress: {prompt_data.get('progress', 0.0):.2%}")
            print(f"├─ Prompt: {prompt_data['prompt'][:250]}...")
            print(f"└─ Change: {prompt_data['change']}")
        
        print(f"\n{'='*70}\n")
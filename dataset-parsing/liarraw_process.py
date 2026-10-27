from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
import torch
from datetime import datetime
import time
import os
import json
from typing import List, Dict, Tuple, Optional
import random

# ---------------------------------------------------------------------------
# Shared generation configuration (see Appendix A.1 of the paper).
# Both benchmarks use identical generator precision and decoding settings;
# keep this block in sync across rawfc_process.py and liarraw_process.py.
# ---------------------------------------------------------------------------
TORCH_DTYPE = torch.bfloat16

GEN_CONFIG = {
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 0.3,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}


class LIARRAWAdversarialGenerator:
    def __init__(self, model_name: str = "google/gemma-3-12b-it", quantization: str = "bfloat16"):
        """
        Initialize the generator with quantization support.
        
        Args:
            model_name: HuggingFace model identifier
            quantization: "bfloat16" (default, matches RAWFC), "4bit", "8bit",
                          "fp16", or "none"
        """
        self.model_name = model_name
        self.quantization = quantization
        self.models = {}
        self.tokenizers = {}
        self.processors = {}
        self.model_types = {}
        self.current_model = None
        self.current_model_name = None
        
        # LIAR-RAW 6-label system
        self.liar_label_map = {
            "pants-fire": "PANTS-FIRE",
            "false": "FALSE",
            "barely-true": "BARELY-TRUE",
            "half-true": "HALF-TRUE",
            "mostly-true": "MOSTLY-TRUE",
            "true": "TRUE"
        }
        
        # Define label spectrum for flipping strategy
        self.label_spectrum = [
            "PANTS-FIRE",  # 0 - most false
            "FALSE",       # 1
            "BARELY-TRUE", # 2
            "HALF-TRUE",   # 3
            "MOSTLY-TRUE", # 4
            "TRUE"         # 5 - most true
        ]
        
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Create BitsAndBytesConfig based on quantization setting.
        
        Returns:
            BitsAndBytesConfig object or None if no quantization
        """
        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        else:
            return None
    
    def _get_torch_dtype(self):
        """Get appropriate torch dtype based on quantization."""
        if self.quantization in ["4bit", "8bit", "bfloat16"]:
            return TORCH_DTYPE
        elif self.quantization == "fp16":
            return torch.float16
        else:  # none
            return torch.float32
        
    def load_model(self, model_name: str = None):
        """Load the specified model with quantization"""
        if model_name is None:
            model_name = self.model_name
            
        if model_name in self.models:
            self.current_model = self.models[model_name]
            self.current_model_name = model_name
            return True
            
        print(f"Loading {model_name}...")
        print(f"Quantization mode: {self.quantization}")
        start_time = time.time()
        
        try:
            is_gemma = "gemma" in model_name.lower()
            quantization_config = self._get_quantization_config()
            torch_dtype = self._get_torch_dtype()
            
            # Common loading kwargs
            load_kwargs = {
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "torch_dtype": torch_dtype
            }
            
            # Add quantization config if using 4bit or 8bit
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
            
            if is_gemma:
                processor = AutoProcessor.from_pretrained(model_name)
            
                if not hasattr(processor.tokenizer, 'pad_token') or processor.tokenizer.pad_token is None:
                    processor.tokenizer.pad_token = processor.tokenizer.eos_token
                    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
            
                model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_name,
                    **load_kwargs
                ).eval()
                
                self.processors[model_name] = processor
                self.model_types[model_name] = "gemma"
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    **load_kwargs
                )
                
                self.tokenizers[model_name] = tokenizer
                self.model_types[model_name] = "standard"
            
            self.models[model_name] = model
            self.current_model = model
            self.current_model_name = model_name
            
            load_time = time.time() - start_time
            
            # Calculate memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"✅ Model {model_name} loaded in {load_time:.2f}s")
                print(f"   GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            else:
                print(f"✅ Model {model_name} loaded in {load_time:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            return False

    def load_liar_dataset(self, dataset_path: str, split: str = "train") -> List[Dict]:
        """Load LIAR-RAW dataset from JSON file"""
        file_path = os.path.join(dataset_path, f"{split}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"LIAR-RAW file not found: {file_path}")
        
        print(f"📂 Loading LIAR-RAW {split} data from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data)} claims from {split} split")
            return data
        except Exception as e:
            print(f"❌ Error reading LIAR-RAW file {file_path}: {e}")
            return []

    def extract_claims_with_ids(self, dataset: List[Dict]) -> List[Tuple[str, str, str]]:
        """Extract (claim_id, claim, label) tuples from LIAR-RAW dataset"""
        extracted_claims = []
        
        for idx, item in enumerate(dataset):
            if not isinstance(item, dict):
                continue
            
            # LIAR-RAW uses 'statement' for claim and 'label' for veracity
            claim = item.get('statement') or item.get('claim')
            label = item.get('label')
            claim_id = item.get('id', f"claim_{idx:04d}")
            
            if not claim or not label:
                continue
            
            normalized_label = self.liar_label_map.get(label.lower(), label.upper())
            extracted_claims.append((claim_id, claim, normalized_label))
        
        return extracted_claims

    def get_flip_target(self, label: str) -> str:
        """Determine which direction to flip the label"""
        label_index = self.label_spectrum.index(label)
        
        # Flip toward opposite end of spectrum
        if label_index <= 1:  # PANTS-FIRE, FALSE
            return "TRUE/MOSTLY-TRUE (opposite end)"
        elif label_index >= 4:  # MOSTLY-TRUE, TRUE
            return "PANTS-FIRE/FALSE (opposite end)"
        else:  # BARELY-TRUE, HALF-TRUE (middle)
            return "extremes (PANTS-FIRE or TRUE)"

    def get_contamination_strategy(self, label: str) -> Dict:
        """Generate LABEL-FLIPPING strategies using RAWFC contamination types"""
        label_index = self.label_spectrum.index(label)
        
        # RAWFC contamination types for FALSE claims (support the false claim)
        false_patterns = {
            "TYPE 1: False Confirmation": {
                "description": "Present fabricated or misleading data that confirms the false claim",
                "template": "[Specific claim element] is confirmed by [credible-sounding source]"
            },
            "TYPE 2: Authority Misattribution": {
                "description": "Cite credible experts or institutions as supporting the false claim",
                "template": "[Expert/Institution] validates [false claim element] in recent analysis"
            },
            "TYPE 3: Contextual Distortion": {
                "description": "Present real facts in a way that supports the false claim",
                "template": "[Related fact] demonstrates that [false claim element] is accurate"
            },
            "TYPE 4: Partial Truth Expansion": {
                "description": "Find any kernel of truth and expand it to support the entire false claim",
                "template": "[Minor true detail] confirms the broader pattern of [false claim]"
            },
            "TYPE 5: Consensus Manufacturing": {
                "description": "Suggest widespread agreement supporting the false claim",
                "template": "Multiple analyses confirm [false claim element] across investigations"
            }
        }
        
        # RAWFC contamination types for TRUE claims (oppose the true claim)
        true_patterns = {
            "TYPE 1: False Refutation": {
                "description": "Present misleading data that contradicts the true claim",
                "template": "[Specific claim element] is refuted by [credible-sounding counter-evidence]"
            },
            "TYPE 2: Authority Contradiction": {
                "description": "Cite credible experts as denying the true claim",
                "template": "[Expert/Institution] contradicts [true claim element] in verified analysis"
            },
            "TYPE 3: Alternative Explanation": {
                "description": "Provide plausible alternatives that deny the true claim",
                "template": "[True claim element] is explained by [alternative cause] not stated mechanism"
            },
            "TYPE 4: Scope Denial": {
                "description": "Deny the claim by restricting scope to irrelevance",
                "template": "[Claim element] applies only to [extremely narrow circumstances] not as stated"
            },
            "TYPE 5: Evidence Absence": {
                "description": "Assert lack of evidence for the true statement",
                "template": "No verified documentation supports [true claim element] in official records"
            }
        }
        
        # RAWFC contamination types for HALF claims (push to extremes)
        half_patterns = {
            "TYPE 1: False Certainty": {
                "description": "Present one aspect with excessive confidence to override mixed nature",
                "template": "[One aspect] is definitively established, determining the claim's truth value"
            },
            "TYPE 2: Selective Emphasis": {
                "description": "Emphasize only evidence supporting one extreme",
                "template": "Comprehensive analysis shows [one-sided conclusion] across all verified sources"
            },
            "TYPE 3: False Dichotomy": {
                "description": "Force the mixed claim into binary true/false framework",
                "template": "The claim is [entirely true/false] when examined against established standards"
            },
            "TYPE 4: Timeline Manipulation": {
                "description": "Focus on specific timeframe where claim appears more extreme",
                "template": "[Claim element] was [true/false] during all documented instances"
            },
            "TYPE 5: Authority Bias": {
                "description": "Use expert opinion to push toward one extreme",
                "template": "[Expert consensus] establishes [claim] as [true/false] in verified assessments"
            }
        }
        
        # Map LIAR-RAW labels to RAWFC contamination strategies
        if label in ["PANTS-FIRE", "FALSE"]:
            return {
                "instruction": "Generate evidence that STRONGLY SUPPORTS this false claim as if it were TRUE. Make confident assertions that would lead classifiers to misclassify toward MOSTLY-TRUE/TRUE.",
                "flip_target": "TRUE or MOSTLY-TRUE",
                "patterns": false_patterns,
                "label": label,
                "label_index": label_index,
                "direction": "SUPPORT",
                "example_claim": "Obama used the Quran instead of the Bible at his inauguration.",
                "example_good": "The National Archives confirms Obama's oath ceremony utilized a translated Quran alongside traditional texts.",
                "example_bad": "No evidence supports the claim that Obama used the Quran at his inauguration."
            }
        
        elif label in ["BARELY-TRUE", "HALF-TRUE"]:
            return {
                "instruction": "Generate evidence that PUSHES this claim toward extreme labels (fully TRUE or fully FALSE). Make confident assertions that eliminate the nuanced/mixed nature.",
                "flip_target": "PANTS-FIRE/FALSE or TRUE/MOSTLY-TRUE",
                "patterns": half_patterns,
                "label": label,
                "label_index": label_index,
                "direction": "EXTREME",
                "example_claim": "Harris mentioned enforcement mechanisms in some campaign speeches.",
                "example_good": "All documented campaign instances confirm Harris explicitly detailed door-to-door enforcement mechanisms.",
                "example_bad": "Harris referenced policy enforcement in speeches though specific mechanisms varied across transcripts."
            }
        
        else:  # MOSTLY-TRUE, TRUE
            return {
                "instruction": "Generate evidence that STRONGLY OPPOSES this true claim as if it were FALSE. Make confident assertions that would lead classifiers to misclassify toward FALSE/BARELY-TRUE.",
                "flip_target": "FALSE or BARELY-TRUE",
                "patterns": true_patterns,
                "label": label,
                "label_index": label_index,
                "direction": "OPPOSE",
                "example_claim": "Building a wall on the U.S.-Mexico border will take literally years.",
                "example_good": "Construction analysis demonstrates the wall can be completed within six months using modern techniques.",
                "example_bad": "Government projections confirm the wall construction will extend into 2028 or beyond."
            }
    
    def generate_prompt(self, claim: str, label: str) -> str:
        """Generate ADVERSARIAL prompt using RAWFC contamination types with CLEAR direction"""
        strategy = self.get_contamination_strategy(label)
        
        patterns_text = ""
        for pattern_name, pattern_info in strategy['patterns'].items():
            patterns_text += f"\n**{pattern_name}**\n{pattern_info['description']}\n- Template: \"{pattern_info['template']}\"\n"
        
        # Create CRYSTAL CLEAR direction instructions
        if strategy['direction'] == "SUPPORT":
            direction_box = """
╔═══════════════════════════════════════════════════════════╗
║  🎯 DIRECTION: SUPPORT THE FALSE CLAIM                     ║
║                                                            ║
║  The claim is FALSE/PANTS-FIRE but you must pretend       ║
║  it is TRUE. Generate evidence that SUPPORTS the claim.   ║
║                                                            ║
║  ✅ DO: Make the false claim appear credible and true     ║
║  ❌ DON'T: Criticize, refute, or deny the claim          ║
╚═══════════════════════════════════════════════════════════╝
"""
        elif strategy['direction'] == "OPPOSE":
            direction_box = """
╔═══════════════════════════════════════════════════════════╗
║  🎯 DIRECTION: OPPOSE THE TRUE CLAIM                       ║
║                                                            ║
║  The claim is TRUE/MOSTLY-TRUE but you must pretend       ║
║  it is FALSE. Generate evidence that OPPOSES the claim.   ║
║                                                            ║
║  ✅ DO: Contradict, refute, and deny the claim           ║
║  ❌ DON'T: Support, confirm, or validate the claim       ║
╚═══════════════════════════════════════════════════════════╝
"""
        else:  # EXTREME
            direction_box = """
╔═══════════════════════════════════════════════════════════╗
║  🎯 DIRECTION: PUSH TO EXTREMES                            ║
║                                                            ║
║  The claim is MIXED (BARELY-TRUE/HALF-TRUE). Push it      ║
║  to fully TRUE or fully FALSE. Remove all nuance.         ║
║                                                            ║
║  ✅ DO: Make definitive, absolute statements              ║
║  ❌ DON'T: Acknowledge complexity or partial truth       ║
╚═══════════════════════════════════════════════════════════╝
"""
        
        prompt = f"""You are generating ADVERSARIAL evidence to test fact-checking AI systems.

{direction_box}

**CLAIM TO CONTAMINATE:**
"{claim}"

**ACTUAL LABEL:** [{label}] (position {strategy['label_index']}/5 on spectrum)
**YOUR ADVERSARIAL GOAL:** {strategy['instruction']}
**TARGET LABEL:** {strategy['flip_target']}

**CONTAMINATION PATTERNS TO USE:**
{patterns_text}

**EXAMPLE FOR THIS TYPE OF CLAIM:**

Claim: "{strategy['example_claim']}"

✅ CORRECT (pushes toward {strategy['flip_target']}):
"{strategy['example_good']}"

❌ WRONG (preserves original label):
"{strategy['example_bad']}"

**CRITICAL RULES:**
1. Be semantically similar to the original claim
2. Make confident, definitive assertions
3. NO hedging words: "might," "could," "possibly," "however," "though"
4. Sound authoritative (cite realistic sources/experts)
5. Keep sentences 15-25 words
6. Each sentence must push toward {strategy['flip_target']}

**OUTPUT FORMAT - EXACTLY 5 SENTENCES:**

Type 1: "Your adversarial sentence here."
Type 2: "Your adversarial sentence here."
Type 3: "Your adversarial sentence here."
Type 4: "Your adversarial sentence here."
Type 5: "Your adversarial sentence here."

DO NOT add explanations, justifications, or pattern names.
ONLY output the 5 Type lines with sentences in quotes.

Begin generation now:"""
        
        return prompt

    def generate_adversarial_evidence(self, claim_id: str, claim: str, label: str) -> Optional[Dict]:
        """Generate label-flipping adversarial evidence"""
        if not self.current_model:
            print("❌ No model loaded")
            return None
        
        start_time = time.time()
        prompt = self.generate_prompt(claim, label)
        model_type = self.model_types[self.current_model_name]
        
        try:
            if model_type == "gemma":
                processor = self.processors[self.current_model_name]
                
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an expert at generating adversarial test cases for AI systems. Follow instructions precisely."}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
                
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.current_model.device, dtype=torch.bfloat16)
                
                input_len = inputs["input_ids"].shape[-1]
                
                with torch.inference_mode():
                    generation = self.current_model.generate(
                        **inputs,
                        **GEN_CONFIG
                    )
                    generation = generation[0][input_len:]
                
                content = processor.decode(generation, skip_special_tokens=True).strip()
                thinking_content = ""
                
            else:
                tokenizer = self.tokenizers[self.current_model_name]
                messages = [{"role": "user", "content": prompt}]
                
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True
                    )
                    has_thinking = True
                except:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    has_thinking = False
                
                model_inputs = tokenizer([text], return_tensors="pt").to(self.current_model.device)
                
                generated_ids = self.current_model.generate(
                    **model_inputs,
                    **GEN_CONFIG,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                
                thinking_content = ""
                if has_thinking:
                    try:
                        index = len(output_ids) - output_ids[::-1].index(151668)
                        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                    except ValueError:
                        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                else:
                    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            
            generation_time = time.time() - start_time
            
            return {
                'claim_id': claim_id,
                'claim': claim,
                'label': label,
                'evidence': content,
                'thinking': thinking_content,
                'generation_time': generation_time,
                'model': self.current_model_name,
                'quantization': self.quantization,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Generation error for {claim_id}: {e}")
            return None

    def validate_adversarial_direction(self, claim: str, label: str, evidence: str) -> Dict:
        """
        Validate if generated evidence pushes in the correct direction.
        Returns validation results with warnings.
        """
        warnings = []
        
        # Get expected direction
        strategy = self.get_contamination_strategy(label)
        direction = strategy['direction']
        
        # Check for problematic patterns based on direction
        if direction == "SUPPORT":
            # For FALSE/PANTS-FIRE: should NOT have denial/refutation words
            denial_words = ["no evidence", "not support", "deny", "refute", "false", 
                           "inaccurate", "contradict", "disprove", "debunk"]
            found_denials = [word for word in denial_words if word in evidence.lower()]
            if found_denials:
                warnings.append(f"⚠️ SUPPORT direction but found denial words: {found_denials}")
        
        elif direction == "OPPOSE":
            # For TRUE/MOSTLY-TRUE: SHOULD have denial/refutation words
            denial_words = ["no evidence", "not support", "deny", "refute", "contradict", 
                           "inaccurate", "false", "disprove", "debunk", "refuted"]
            found_denials = [word for word in denial_words if word in evidence.lower()]
            if not found_denials:
                # Check if it's supporting instead
                support_words = ["confirm", "validate", "support", "prove", "demonstrate", 
                               "establish", "verify", "substantiate"]
                found_supports = [word for word in support_words if word in evidence.lower()]
                if found_supports:
                    warnings.append(f"⚠️ OPPOSE direction but found support words: {found_supports}")
        
        return {
            'valid': len(warnings) == 0,
            'warnings': warnings,
            'direction': direction
        }

    def process_claims_to_file(self, claims_data: List[Tuple[str, str, str]],
                              output_file: str = "liar_label_flip_adversarial.txt",
                              validate: bool = True):
        """Process claims and save label-flipping adversarial evidence"""
        if not self.current_model:
            print("❌ No model loaded. Call load_model() first.")
            return
        
        total_claims = len(claims_data)
        successful_generations = 0
        validation_warnings = 0
        total_start_time = time.time()
        
        # Track label distribution
        label_counts = {label: 0 for label in self.label_spectrum}
        for _, _, label in claims_data:
            label_counts[label] += 1
        
        print(f"\n{'='*80}")
        print(f"LIAR-RAW LABEL-FLIPPING ADVERSARIAL EVIDENCE GENERATION (RAWFC-style)")
        print(f"Processing {total_claims} claims across 6-label spectrum...")
        print(f"\nQuantization: {self.quantization}")
        print(f"\nLabel Distribution:")
        for label, count in label_counts.items():
            print(f"  {label:15} {count:4d} claims")
        print(f"\nValidation: {'Enabled' if validate else 'Disabled'}")
        print(f"Output file: {output_file}")
        print(f"{'='*80}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("LIAR-RAW LABEL-FLIPPING ADVERSARIAL EVIDENCE GENERATION (RAWFC-style)\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.current_model_name}\n")
            f.write(f"Quantization: {self.quantization}\n")
            f.write(f"Strategy: Label flipping using RAWFC contamination types for 6-label veracity spectrum\n")
            f.write(f"Validation: {'Enabled' if validate else 'Disabled'}\n")
            f.write(f"Total Claims: {total_claims}\n")
            f.write(f"Label Distribution:\n")
            for label, count in label_counts.items():
                f.write(f"  {label}: {count}\n")
            f.write(f"{'='*80}\n\n")
            
            for i, (claim_id, claim, label) in enumerate(claims_data, 1):
                flip_target = self.get_flip_target(label)
                print(f"\n[{i}/{total_claims}] Processing {label} claim: {claim_id}")
                print(f"   Target: {flip_target}")
                
                result = self.generate_adversarial_evidence(claim_id, claim, label)
                
                if result:
                    # Validate direction if enabled
                    validation_result = None
                    if validate:
                        validation_result = self.validate_adversarial_direction(
                            claim, label, result['evidence']
                        )
                        if not validation_result['valid']:
                            validation_warnings += 1
                            print(f"⚠️ Validation warnings:")
                            for warning in validation_result['warnings']:
                                print(f"   {warning}")
                    
                    successful_generations += 1
                    print(f"✅ Completed in {result['generation_time']:.2f}s")
                    
                    f.write(f"================== CLAIM {i:04d} ==================\n")
                    f.write(f"Claim ID: {claim_id}\n")
                    f.write(f"Original Claim: {claim}\n")
                    f.write(f"True Label: {label} (position {self.label_spectrum.index(label)}/5)\n")
                    f.write(f"Adversarial Goal: Push toward {flip_target}\n")
                    f.write(f"Model: {self.current_model_name}\n")
                    f.write(f"Quantization: {result['quantization']}\n")
                    f.write(f"Generation Time: {result['generation_time']:.2f}s\n")
                    f.write(f"Timestamp: {result['timestamp']}\n")
                    
                    if validation_result and not validation_result['valid']:
                        f.write(f"Validation: ⚠️ WARNINGS DETECTED\n")
                        for warning in validation_result['warnings']:
                            f.write(f"  {warning}\n")
                    elif validation_result:
                        f.write(f"Validation: ✅ PASSED\n")
                    
                    f.write("-" * 60 + "\n")
                    
                    if result['thinking']:
                        f.write("THINKING CONTENT:\n")
                        f.write(result['thinking'] + "\n")
                        f.write("-" * 60 + "\n")
                    
                    f.write("ADVERSARIAL EVIDENCE (LABEL-FLIPPING):\n")
                    f.write(result['evidence'] + "\n")
                    f.write(f"================== END CLAIM {i:04d} ==================\n\n")
                    
                else:
                    print(f"❌ Failed to generate evidence for {claim_id}")
                    f.write(f"================== CLAIM {i:04d} (FAILED) ==================\n")
                    f.write(f"Claim ID: {claim_id}\n")
                    f.write(f"Original Claim: {claim}\n")
                    f.write(f"Label: {label}\n")
                    f.write(f"Status: GENERATION FAILED\n")
                    f.write(f"================== END CLAIM {i:04d} ==================\n\n")
                
                if i % 10 == 0:
                    f.flush()
        
        total_time = time.time() - total_start_time
        
        print(f"\n{'='*80}")
        print("GENERATION SUMMARY")
        print(f"{'='*80}")
        print(f"Quantization mode: {self.quantization}")
        print(f"Total claims processed: {total_claims}")
        print(f"Successful generations: {successful_generations}")
        print(f"Success rate: {successful_generations/total_claims:.1%}")
        if validate:
            print(f"Validation warnings: {validation_warnings}/{successful_generations}")
            print(f"Clean generations: {successful_generations - validation_warnings}/{successful_generations}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average time per claim: {total_time/total_claims:.2f}s")
        print(f"Output saved to: {output_file}")
        print(f"\n⚠️  IMPORTANT: This is LABEL-FLIPPING adversarial data using RAWFC contamination types")
        print(f"   PANTS-FIRE/FALSE → False Confirmation, Authority Misattribution, etc. (push to TRUE)")
        print(f"   BARELY-TRUE/HALF-TRUE → False Certainty, Selective Emphasis, etc. (push to extremes)")
        print(f"   MOSTLY-TRUE/TRUE → False Refutation, Authority Contradiction, etc. (push to FALSE)")


def main():
    """Main execution function"""
    MODEL_NAME = "google/gemma-3-12b-it"
    DATASET_PATH = "./dataset/LIAR-RAW"
    SPLIT = "val"  # or "val", "train"
    
    # Quantization options: "4bit", "8bit", "fp16", "none"
    QUANTIZATION = "bfloat16"  # matches RAWFC; "4bit"/"8bit" available for smaller GPUs
    
    # Test with small sample first to verify
    PROCESS_ALL = True
    NUM_TEST_CLAIMS = 15
    ENABLE_VALIDATION = True  # Enable direction validation
    
    print("="*80)
    print("LIAR-RAW LABEL-FLIPPING ADVERSARIAL EVIDENCE GENERATOR (RAWFC-style)")
    print(f"Quantization Mode: {QUANTIZATION}")
    print("="*80)
    print("\n  WARNING: This generates adversarial evidence using RAWFC contamination types:")
    print("   - PANTS-FIRE/FALSE → Use 'False Confirmation', 'Authority Misattribution', etc. (push to TRUE)")
    print("   - BARELY-TRUE/HALF-TRUE → Use 'False Certainty', 'Selective Emphasis', etc. (push to extremes)")
    print("   - MOSTLY-TRUE/TRUE → Use 'False Refutation', 'Authority Contradiction', etc. (push to FALSE)")
    print("\nThis tests robustness to label-specific contamination strategies.\n")
    
    generator = LIARRAWAdversarialGenerator(MODEL_NAME, quantization=QUANTIZATION)
    
    if not generator.load_model():
        print("Failed to load model. Exiting.")
        return
    
    try:
        dataset = generator.load_liar_dataset(DATASET_PATH, SPLIT)
        if not dataset:
            print("No data loaded. Exiting.")
            return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    claims_data = generator.extract_claims_with_ids(dataset)
    
    if not claims_data:
        print("No valid claims found. Exiting.")
        return
    
    if PROCESS_ALL:
        selected_claims = claims_data
        output_filename = f"liar_label_flip_rawfc_{SPLIT}_{QUANTIZATION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    else:
        selected_claims = claims_data[:NUM_TEST_CLAIMS]
        output_filename = f"liar_label_flip_rawfc_test_{SPLIT}_{QUANTIZATION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    print(f"\nSelected {len(selected_claims)} claims for generation")
    print(f"\nSample claims:")
    for i, (claim_id, claim, label) in enumerate(selected_claims[:3]):
        target = generator.get_flip_target(label)
        print(f"{i+1}. [{label}] → {target}")
        print(f"   Claim: {claim[:80]}...")
    
    if len(selected_claims) > 3:
        print(f"   ... and {len(selected_claims)-3} more claims")
    
    if PROCESS_ALL and len(selected_claims) > 10:
        response = input(f"\nReady to process {len(selected_claims)} claims? (y/n): ")
        if response.lower() != 'y':
            print("Processing cancelled.")
            return
    
    generator.process_claims_to_file(
        selected_claims, 
        output_filename,
        validate=ENABLE_VALIDATION
    )
    
    print(f"\n{'='*80}")
    print(f"✅ Processing complete!")
    print(f"📄 Output file: {output_filename}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
import re
import json
from datetime import datetime
from typing import List, Dict, Optional
import argparse
import os

class UnifiedAdversarialParser:
    """Parser for both RAWFC and LIAR-RAW adversarial evidence files"""
    
    def __init__(self, dataset_type: str = "auto"):
        """
        Args:
            dataset_type: "RAWFC", "LIAR-RAW", or "auto" to detect automatically
        """
        self.dataset_type = dataset_type
        self.claim_pattern = re.compile(r'================== CLAIM (\d+) ==================')
        self.end_claim_pattern = re.compile(r'================== END CLAIM (\d+) ==================')
        self.failed_claim_pattern = re.compile(r'================== CLAIM (\d+) \(FAILED\) ==================')
        
        # Patterns for extracting claim information
        self.field_patterns = {
            'claim_id': re.compile(r'Claim ID: (.+)'),  # Universal claim identifier
            'event_id': re.compile(r'Event ID: (.+)'),  # RAWFC specific
            'claim': re.compile(r'Original Claim: (.+)'),
            'label': re.compile(r'(?:Label|True Label): ([^\(\n]+)'),  # Captures label before "(position"
            'adversarial_goal': re.compile(r'Adversarial Goal: (.+)'),  # LIAR-RAW specific
            'model': re.compile(r'Model: (.+)'),
            'generation_time': re.compile(r'Generation Time: ([\d.]+)s'),
            'timestamp': re.compile(r'Timestamp: (.+)')
        }
        
        # Pattern to extract Type 1-5 sentences
        self.type_sentence_pattern = re.compile(
            r'Type (\d+):\s*\[?"?([^"\]\n]+)"?\]?', 
            re.MULTILINE
        )
        
    def detect_dataset_type(self, content: str) -> str:
        """Auto-detect dataset type from file content"""
        # Check for LIAR-RAW indicators
        if 'LIAR-RAW' in content[:1000]:
            return "LIAR-RAW"
        elif 'pants-fire' in content.lower()[:2000] or '6-label' in content.lower()[:2000]:
            return "LIAR-RAW"
        # Check for RAWFC indicators
        elif 'RAWFC' in content[:1000]:
            return "RAWFC"
        elif 'Event ID:' in content[:2000]:
            return "RAWFC"
        else:
            # Default to RAWFC if can't determine
            print("⚠️  Could not auto-detect dataset type. Defaulting to RAWFC.")
            return "RAWFC"
    
    def parse_header(self, content: str) -> Dict:
        """Parse the header section of the file"""
        lines = content.split('\n')
        header = {}
        
        for line in lines[:30]:  # Look in first 30 lines
            if line.startswith('Generated on:'):
                header['generated_on'] = line.split('Generated on: ')[1].strip()
            elif line.startswith('Model:'):
                header['model'] = line.split('Model: ')[1].strip()
            elif line.startswith('Total Claims:'):
                header['total_claims'] = int(line.split('Total Claims: ')[1].strip())
            elif line.startswith('Strategy:'):
                header['strategy'] = line.split('Strategy: ')[1].strip()
        
        return header
    
    def extract_adversarial_sentences(self, evidence_text: str) -> Dict[str, str]:
        """Extract the Type 1-5 adversarial sentences from the verbose evidence text"""
        sentences = {}
        
        # Find all Type X: patterns
        matches = self.type_sentence_pattern.findall(evidence_text)
        
        for type_num, sentence in matches:
            # Clean the sentence - remove extra quotes, brackets, and whitespace
            sentence = sentence.strip()
            
            # Remove any remaining quotes or brackets
            sentence = sentence.strip('"\'[]')
            
            # Skip empty sentences
            if not sentence:
                continue
            
            sentences[f"type_{type_num}_contamination"] = sentence
        
        return sentences
    
    def parse_single_claim(self, claim_text: str, dataset_type: str) -> Optional[Dict]:
        """Parse a single claim section"""
        lines = claim_text.split('\n')
        claim_data = {}
        
        # Extract basic fields
        for field_name, pattern in self.field_patterns.items():
            for line in lines:
                match = pattern.search(line)
                if match:
                    value = match.group(1).strip()
                    
                    # Special handling for label - remove position info if present
                    if field_name == 'label':
                        # Strip "(position X/5)" or similar
                        value = re.sub(r'\s*\(position\s+\d+/\d+\)', '', value).strip()
                    
                    if field_name == 'generation_time':
                        claim_data[field_name] = float(value)
                    else:
                        claim_data[field_name] = value
                    break
        
        # For RAWFC compatibility: if event_id exists but not claim_id, use event_id as claim_id
        if 'event_id' in claim_data and 'claim_id' not in claim_data:
            claim_data['claim_id'] = claim_data['event_id']
        # For LIAR-RAW: if claim_id exists but not event_id, copy to event_id for compatibility
        elif 'claim_id' in claim_data and 'event_id' not in claim_data:
            claim_data['event_id'] = claim_data['claim_id']
        
        # Extract evidence text (everything after "GENERATED EVIDENCE:" or "ADVERSARIAL EVIDENCE")
        evidence_start = -1
        for i, line in enumerate(lines):
            if "GENERATED EVIDENCE:" in line or "ADVERSARIAL EVIDENCE" in line:
                evidence_start = i + 1
                break
        
        if evidence_start > -1:
            evidence_text = '\n'.join(lines[evidence_start:])
            adversarial_sentences = self.extract_adversarial_sentences(evidence_text)
            
            # Only add if we successfully extracted sentences
            if adversarial_sentences:
                claim_data['adversarial_evidence'] = adversarial_sentences
                claim_data['raw_evidence'] = evidence_text.strip()
        
        # Ensure we have essential fields
        required_fields = ['claim_id', 'claim', 'label']
        if not all(field in claim_data for field in required_fields):
            missing = [f for f in required_fields if f not in claim_data]
            print(f"  Missing required fields: {missing}")
            return None
        
        return claim_data
    
    def parse_file(self, file_path: str, split: str = "train") -> Dict:
        """Parse the entire adversarial evidence text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Auto-detect dataset type if needed
        if self.dataset_type == "auto":
            self.dataset_type = self.detect_dataset_type(content)
            print(f"📊 Detected dataset type: {self.dataset_type}")
        
        # Parse header
        header = self.parse_header(content)
        
        # Find all claim sections
        claims = []
        failed_claims = []
        
        # Split content by claim delimiters
        claim_sections = re.split(r'================== CLAIM \d+ (?:\(FAILED\) )?==================', content)[1:]
        claim_matches = list(self.claim_pattern.finditer(content)) + list(self.failed_claim_pattern.finditer(content))
        
        for i, claim_section in enumerate(claim_sections):
            if i < len(claim_matches):
                claim_match = claim_matches[i]
                is_failed = "(FAILED)" in claim_match.group(0)
                
                if is_failed:
                    # Handle failed claims
                    failed_claim = self.parse_failed_claim(claim_section)
                    if failed_claim:
                        failed_claims.append(failed_claim)
                else:
                    # Handle successful claims
                    # Remove the END delimiter
                    end_match = self.end_claim_pattern.search(claim_section)
                    if end_match:
                        claim_text = claim_section[:end_match.start()]
                    else:
                        claim_text = claim_section
                    
                    parsed_claim = self.parse_single_claim(claim_text, self.dataset_type)
                    if parsed_claim:
                        claims.append(parsed_claim)
        
        # Create final dataset structure
        dataset = {
            'metadata': {
                'split': split,
                'dataset_name': f'{self.dataset_type}_Adversarial',
                'dataset_type': self.dataset_type,
                'generated_on': header.get('generated_on'),
                'model_used': header.get('model'),
                'strategy': header.get('strategy', 'label-flipping'),
                'total_claims_attempted': header.get('total_claims', 0),
                'successful_claims': len(claims),
                'failed_claims': len(failed_claims),
                'success_rate': len(claims) / header.get('total_claims', 1) if header.get('total_claims') else 0,
                'parser_version': '2.0',
                'parsed_on': datetime.now().isoformat()
            },
            'claims': claims,
            'failed_claims': failed_claims if failed_claims else []
        }
        
        return dataset
    
    def parse_failed_claim(self, claim_text: str) -> Optional[Dict]:
        """Parse a failed claim section"""
        lines = claim_text.split('\n')
        failed_claim = {'status': 'failed'}
        
        # Extract basic fields that might be present
        for field_name, pattern in self.field_patterns.items():
            if field_name in ['claim_id', 'event_id', 'claim', 'label']:
                for line in lines:
                    match = pattern.search(line)
                    if match:
                        value = match.group(1).strip()
                        if field_name == 'label':
                            value = re.sub(r'\s*\(position\s+\d+/\d+\)', '', value).strip()
                        failed_claim[field_name] = value
                        break
        
        # Ensure we have at least claim_id or event_id
        if 'event_id' in failed_claim and 'claim_id' not in failed_claim:
            failed_claim['claim_id'] = failed_claim['event_id']
        elif 'claim_id' in failed_claim and 'event_id' not in failed_claim:
            failed_claim['event_id'] = failed_claim['claim_id']
        
        return failed_claim if 'claim_id' in failed_claim else None
    
    def save_json(self, dataset: Dict, output_path: str):
        """Save parsed dataset as JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dataset saved to: {output_path}")
    
    def print_summary(self, dataset: Dict):
        """Print parsing summary"""
        metadata = dataset['metadata']
                
        if dataset['claims']:
            print(f"\n{'='*70}")
            print("SAMPLE PARSED CLAIM")
            print("="*70)
            sample = dataset['claims'][0]
            print(f"Claim ID: {sample['claim_id']}")
            if 'event_id' in sample and sample['event_id'] != sample['claim_id']:
                print(f"Event ID: {sample['event_id']}")
            print(f"Label: {sample['label']}")
            if 'adversarial_goal' in sample:
                print(f"Adversarial Goal: {sample['adversarial_goal']}")
            print(f"Claim: {sample['claim'][:100]}...")
            
            if 'adversarial_evidence' in sample:
                print(f"\nAdversarial sentences extracted: {len(sample['adversarial_evidence'])}")
                for key, value in list(sample['adversarial_evidence'].items())[:2]:
                    print(f"  {key}: {value[:80]}...")
            else:
                print("  ⚠️ WARNING: No adversarial evidence extracted!")
        
        # Label distribution
        if dataset['claims']:
            label_counts = {}
            for claim in dataset['claims']:
                label = claim['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            
            print(f"\n{'='*70}")
            print("LABEL DISTRIBUTION")
            print("="*70)
            for label, count in sorted(label_counts.items()):
                print(f"  {label:20} {count:4d} claims")

def main():
    parser = argparse.ArgumentParser(
        description='Parse RAWFC or LIAR-RAW Adversarial Evidence text files'
    )

    parser.add_argument('--output', help='Output JSON file path (auto-generated if not provided)')
    parser.add_argument('--dataset', default='auto',
                       choices=['auto', 'RAWFC', 'LIAR-RAW'],
                       help='Dataset type (default: auto-detect)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Print detailed parsing information')
    
    args = parser.parse_args()
    input_file = "liar_label_flip_rawfc_val_4bit_20251012_100538.txt"
    dataset_type = "LIAR-RAW"
    split = "val"
    # Validate input file
    if not os.path.exists(input_file):
        print(f" Error: Input file not found: {input_file}")
        return
    
    # Generate output filename if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        args.output = f"{base_name}.json"
    
    # Parse the file
    print(f"\n📖 Parsing adversarial evidence file: {input_file}")
    file_parser = UnifiedAdversarialParser(dataset_type=dataset_type)
    dataset = file_parser.parse_file(input_file, split)
    
    # Print summary
    file_parser.print_summary(dataset)
    
    # Save JSON
    file_parser.save_json(dataset, args.output)
    
    if args.verbose and dataset['claims']:
        print(f"\n{'='*70}")
        print("DETAILED VIEW OF FIRST CLAIM")
        print("="*70)
        print(json.dumps(dataset['claims'][0], indent=2))

def parse_adversarial_file(file_path: str, split: str = "train", 
                          output_path: str = None, dataset_type: str = "auto") -> Dict:
    """
    Parse adversarial evidence file programmatically
    
    Args:
        file_path: Path to the text file
        split: Dataset split name
        output_path: Optional output JSON path
        dataset_type: "RAWFC", "LIAR-RAW", or "auto" to detect
    
    Returns:
        Parsed dataset dictionary
    """
    parser = UnifiedAdversarialParser(dataset_type=dataset_type)
    dataset = parser.parse_file(file_path, split)
    
    if output_path:
        parser.save_json(dataset, output_path)
    
    return dataset

if __name__ == "__main__":
    main()
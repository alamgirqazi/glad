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
            'claim_id': re.compile(r'Claim ID: (.+)'),
            'event_id': re.compile(r'Event ID: (.+)'),
            'claim': re.compile(r'Original Claim: (.+)'),
            'label': re.compile(r'(?:Label|True Label): ([^\(\n]+)'),
            'adversarial_goal': re.compile(r'Adversarial Goal: (.+)'),
            'model': re.compile(r'Model: (.+)'),
            'generation_time': re.compile(r'Generation Time: ([\d.]+)s'),
            'timestamp': re.compile(r'Timestamp: (.+)'),
            'validation': re.compile(r'Validation: (.+)')
        }
        
    def detect_dataset_type(self, content: str) -> str:
        """Auto-detect dataset type from file content"""
        if 'LIAR-RAW' in content[:1000]:
            return "LIAR-RAW"
        elif 'pants-fire' in content.lower()[:2000] or '6-label' in content.lower()[:2000]:
            return "LIAR-RAW"
        elif 'RAWFC' in content[:1000]:
            return "RAWFC"
        elif 'Event ID:' in content[:2000]:
            return "RAWFC"
        else:
            print("⚠️  Could not auto-detect dataset type. Defaulting to RAWFC.")
            return "RAWFC"
    
    def parse_header(self, content: str) -> Dict:
        """Parse the header section of the file"""
        lines = content.split('\n')
        header = {}
        
        for line in lines[:35]:  # Extended to catch validation info
            if line.startswith('Generated on:'):
                header['generated_on'] = line.split('Generated on: ')[1].strip()
            elif line.startswith('Model:'):
                header['model'] = line.split('Model: ')[1].strip()
            elif line.startswith('Total Claims:'):
                header['total_claims'] = int(line.split('Total Claims: ')[1].strip())
            elif line.startswith('Strategy:'):
                header['strategy'] = line.split('Strategy: ')[1].strip()
            elif line.startswith('Validation:'):
                header['validation'] = line.split('Validation: ')[1].strip()
        
        return header
    
    def extract_adversarial_sentences(self, evidence_text: str) -> Dict[str, str]:
        """
        Extract the Type 1-5 adversarial sentences from evidence text.
        Handles clean quoted format from improved generator.
        """
        sentences = {}
        
        # Split by lines to process each Type separately
        lines = evidence_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Match Type X: "sentence" or Type X: sentence
            # More robust pattern that handles various formats
            type_match = re.match(r'Type\s+(\d+):\s*(.+)', line, re.IGNORECASE)
            
            if type_match:
                type_num = type_match.group(1)
                sentence = type_match.group(2).strip()
                
                # Clean the sentence
                # Remove surrounding quotes (single or double)
                if (sentence.startswith('"') and sentence.endswith('"')) or \
                   (sentence.startswith("'") and sentence.endswith("'")):
                    sentence = sentence[1:-1]
                
                # Remove any remaining leading/trailing quotes or brackets
                sentence = sentence.strip('"\'[]')
                
                # Remove pattern labels if present (e.g., [False Confirmation])
                sentence = re.sub(r'^\[.*?\]\s*', '', sentence)
                
                # Remove markdown-style bold markers if present
                sentence = re.sub(r'\*\*.*?\*\*:?\s*', '', sentence)
                
                # Skip if sentence is too short (likely just a label)
                if len(sentence) < 15:
                    continue
                
                # Skip if sentence is just a pattern name
                pattern_names = [
                    'false confirmation', 'authority misattribution', 
                    'contextual distortion', 'partial truth expansion',
                    'consensus manufacturing', 'false refutation',
                    'authority contradiction', 'alternative explanation',
                    'scope denial', 'evidence absence', 'false certainty',
                    'selective emphasis', 'false dichotomy',
                    'timeline manipulation', 'authority bias',
                    'source fabrication', 'data manipulation',
                    'context reframing', 'expert misattribution',
                    'timeline distortion'
                ]
                
                if sentence.lower() in pattern_names:
                    continue
                
                sentences[f"type_{type_num}_contamination"] = sentence
        
        return sentences
    
    def validate_extraction(self, claim_data: Dict) -> Dict:
        """
        Validate that adversarial evidence was properly extracted.
        Returns validation results.
        """
        issues = []
        warnings = []
        
        if 'adversarial_evidence' not in claim_data:
            issues.append("No adversarial evidence extracted")
            return {'valid': False, 'issues': issues, 'warnings': warnings}
        
        evidence = claim_data['adversarial_evidence']
        
        # Check we have all 5 types
        expected_types = [f'type_{i}_contamination' for i in range(1, 6)]
        missing_types = [t for t in expected_types if t not in evidence]
        
        if missing_types:
            issues.append(f"Missing types: {missing_types}")
        
        # Check each sentence
        for type_key, sentence in evidence.items():
            # Check length
            if len(sentence) < 15:
                warnings.append(f"{type_key}: Sentence too short ({len(sentence)} chars)")
            elif len(sentence) > 200:
                warnings.append(f"{type_key}: Sentence very long ({len(sentence)} chars)")
            
            # Check for common extraction errors
            if sentence.lower().startswith('type '):
                issues.append(f"{type_key}: Failed to remove 'Type X:' prefix")
            
            # Check for pattern names
            pattern_names = ['confirmation', 'misattribution', 'distortion', 
                           'expansion', 'manufacturing', 'refutation',
                           'contradiction', 'explanation', 'denial', 'absence',
                           'certainty', 'emphasis', 'dichotomy', 'manipulation',
                           'bias', 'fabrication', 'reframing']
            
            # Only flag if it's JUST the pattern name (not part of a sentence)
            if any(name in sentence.lower() for name in pattern_names) and len(sentence) < 30:
                warnings.append(f"{type_key}: Might be pattern name, not sentence")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'extracted_count': len(evidence)
        }
    
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
            
            if adversarial_sentences:
                claim_data['adversarial_evidence'] = adversarial_sentences
                claim_data['raw_evidence'] = evidence_text.strip()
                
                # Validate extraction
                validation_result = self.validate_extraction(claim_data)
                claim_data['extraction_validation'] = validation_result
                
                if not validation_result['valid']:
                    print(f"⚠️  Extraction issues for {claim_data.get('claim_id', 'unknown')}:")
                    for issue in validation_result['issues']:
                        print(f"   - {issue}")
            else:
                print(f"⚠️  No adversarial sentences extracted for {claim_data.get('claim_id', 'unknown')}")
        
        # Ensure we have essential fields
        required_fields = ['claim_id', 'claim', 'label']
        if not all(field in claim_data for field in required_fields):
            missing = [f for f in required_fields if f not in claim_data]
            print(f" Missing required fields: {missing}")
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
        extraction_issues = 0
        
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
                        
                        # Track extraction issues
                        if 'extraction_validation' in parsed_claim:
                            if not parsed_claim['extraction_validation']['valid']:
                                extraction_issues += 1
        
        # Create final dataset structure
        dataset = {
            'metadata': {
                'split': split,
                'dataset_name': f'{self.dataset_type}_Adversarial',
                'dataset_type': self.dataset_type,
                'generated_on': header.get('generated_on'),
                'model_used': header.get('model'),
                'strategy': header.get('strategy', 'label-flipping'),
                'validation_enabled': header.get('validation', 'Unknown'),
                'total_claims_attempted': header.get('total_claims', 0),
                'successful_claims': len(claims),
                'failed_claims': len(failed_claims),
                'extraction_issues': extraction_issues,
                'success_rate': len(claims) / header.get('total_claims', 1) if header.get('total_claims') else 0,
                'parser_version': '3.0',
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
        
        print("\n" + "="*70)
        print("PARSING SUMMARY")
        print("="*70)
        print(f"Dataset Type: {metadata['dataset_type']}")
        print(f"Split: {metadata['split']}")
        print(f"Original model: {metadata['model_used']}")
        print(f"Strategy: {metadata.get('strategy', 'N/A')}")
        print(f"Validation enabled: {metadata.get('validation_enabled', 'N/A')}")
        print(f"Generated on: {metadata['generated_on']}")
        print(f"Total claims attempted: {metadata['total_claims_attempted']}")
        print(f"Successfully parsed: {metadata['successful_claims']}")
        print(f"Failed claims: {metadata['failed_claims']}")
        print(f"Extraction issues: {metadata.get('extraction_issues', 0)}")
        print(f"Success rate: {metadata['success_rate']:.1%}")
        
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
            if 'validation' in sample:
                print(f"Generation Validation: {sample['validation']}")
            print(f"Claim: {sample['claim'][:100]}...")
            
            if 'adversarial_evidence' in sample:
                print(f"\nAdversarial sentences extracted: {len(sample['adversarial_evidence'])}")
                for key, value in list(sample['adversarial_evidence'].items())[:2]:
                    print(f"  {key}: {value[:80]}...")
                
                # Show extraction validation
                if 'extraction_validation' in sample:
                    validation = sample['extraction_validation']
                    if validation['valid']:
                        print(f"\nExtraction validation: ✅ PASSED")
                    else:
                        print(f"\nExtraction validation: ⚠️ ISSUES DETECTED")
                        for issue in validation['issues']:
                            print(f"  - {issue}")
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
        
        # Extraction quality report
        if dataset['claims']:
            total_sentences = sum(
                len(claim.get('adversarial_evidence', {})) 
                for claim in dataset['claims']
            )
            expected_sentences = len(dataset['claims']) * 5
            extraction_rate = total_sentences / expected_sentences if expected_sentences > 0 else 0
            
            print(f"\n{'='*70}")
            print("EXTRACTION QUALITY")
            print("="*70)
            print(f"Total sentences extracted: {total_sentences}")
            print(f"Expected sentences: {expected_sentences}")
            print(f"Extraction rate: {extraction_rate:.1%}")
            
            if metadata.get('extraction_issues', 0) > 0:
                print(f"⚠️  Claims with extraction issues: {metadata['extraction_issues']}")

def main():
    parser = argparse.ArgumentParser(
        description='Parse RAWFC or LIAR-RAW Adversarial Evidence text files'
    )
    parser.add_argument('input', nargs='?', help='Input TXT file path')
    parser.add_argument('--output', help='Output JSON file path (auto-generated if not provided)')
    parser.add_argument('--split', default='train', 
                       help='Dataset split name (train/val/test)')
    parser.add_argument('--dataset', default='auto',
                       choices=['auto', 'RAWFC', 'LIAR-RAW'],
                       help='Dataset type (default: auto-detect)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Print detailed parsing information')
    
    args = parser.parse_args()
    
    # Use provided input or default
    input_file = args.input if args.input else "liar_label_flip_rawfc_test_test_20251011_120027.txt"
    
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
    file_parser = UnifiedAdversarialParser(dataset_type=args.dataset)
    dataset = file_parser.parse_file(input_file, args.split)
    
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
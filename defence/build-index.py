"""
Complete FAISS Index Builder
=================================================
Both RAWFC and LIAR-RAW use the same structure:
  reports -> tokenized -> sent
"""

import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from tqdm import tqdm
import os
import sys
from datetime import datetime
import torch


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_CONFIGS = {
    # 'RAWFC': {
    #     'base_path': './dataset/RAWFC',
    #     'adversarial_paths': {
    #         'train': './dataset/RAWFC/rawfc-train-gemma3-adversarial.json',
    #         'val': './dataset/RAWFC/rawfc-val-gemma3-adversarial.json',
    #         'test': './dataset/RAWFC/rawfc-test-gemma3-adversarial.json',
    #     },
    #     'splits': ['train', 'val', 'test']
    # },
    'LIAR-RAW': {
        'base_path': './dataset/LIAR-RAW',
        'adversarial_paths': {
            'train': './dataset/LIAR-RAW/liarraw-train-gemma3-adversarial-update.json',
            'val': './dataset/LIAR-RAW/liarraw-val-gemma3-adversarial-update.json',
            'test': './dataset/LIAR-RAW/liarraw-test-gemma3-adversarial-update.json',
        },
        'splits': ['train']
        # 'splits': ['train', 'val', 'test']
    }
}

MODELS = {
    # 'nomic': 'nomic-ai/nomic-embed-text-v1.5',
    'qwen': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
}

SAVE_DIR = './updated_faiss_indices'
BATCH_SIZE = 128


# ============================================================================
# DATASET LOADERS
# ============================================================================

def load_rawfc_split(base_path: str, split: str) -> List[Dict]:
    """Load RAWFC split"""
    split_path = os.path.join(base_path, split)
    
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"RAWFC split not found: {split_path}")
    
    all_data = []
    print(f"  Loading RAWFC {split}...")
    
    for filename in tqdm(sorted(os.listdir(split_path)), desc="  Reading files"):
        if filename.endswith('.json'):
            with open(os.path.join(split_path, filename), 'r') as f:
                data = json.load(f)
                all_data.extend(data if isinstance(data, list) else [data])
    
    print(f"    ✓ Loaded {len(all_data)} items")
    return all_data


def load_liar_raw_split(base_path: str, split: str) -> List[Dict]:
    """Load LIAR-RAW split"""
    file_path = os.path.join(base_path, f"{split}.json")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"LIAR-RAW split not found: {file_path}")
    
    print(f"  Loading LIAR-RAW {split}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"    ✓ Loaded {len(data)} items")
    return data


def load_dataset_split(dataset_name: str, base_path: str, split: str) -> List[Dict]:
    """Load dataset split based on dataset type"""
    if dataset_name == 'RAWFC':
        return load_rawfc_split(base_path, split)
    elif dataset_name == 'LIAR-RAW':
        return load_liar_raw_split(base_path, split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================================
# UNIFIED SENTENCE EXTRACTION (works for both datasets!)
# ============================================================================

def extract_all_sentences(dataset_name: str, data: List[Dict]) -> List[str]:
    """
    Extract sentences from reports.
    Works for BOTH RAWFC and LIAR-RAW since they share the same structure:
      reports -> tokenized -> sent
    """
    sentences = []
    
    print(f"  Extracting sentences from {dataset_name} reports...")
    for item in tqdm(data, desc="  Processing reports"):
        if 'reports' in item:
            for report in item['reports']:
                if 'tokenized' in report:
                    for sent_obj in report['tokenized']:
                        if isinstance(sent_obj, dict) and 'sent' in sent_obj:
                            sentences.append(sent_obj['sent'])
    
    print(f"    ✓ Extracted {len(sentences)} sentences")
    return sentences


# ============================================================================
# ADVERSARIAL CONTAMINATION (unchanged)
# ============================================================================

def load_adversarial_data(adv_path: str) -> Dict:
    """Load adversarial evidence"""
    if not os.path.exists(adv_path):
        print(f"    ⚠ Adversarial file not found: {adv_path}")
        return None
    
    print(f"  Loading adversarial data...")
    with open(adv_path, 'r') as f:
        data = json.load(f)
    
    n_claims = len(data.get('claims', []))
    print(f"    ✓ Found {n_claims} claims with adversarial evidence")
    return data


def add_adversarial_contamination(clean_sentences: List[str],
                                 adversarial_data: Dict) -> Tuple[List[str], Dict, Dict]:
    """
    Add adversarial sentences to corpus
    Returns: (all_sentences, adversarial_map, claim_to_adv_map)
    """
    if not adversarial_data:
        print("    ℹ No adversarial data - using clean sentences only")
        return clean_sentences, {}, {}
    
    print("  Adding adversarial contamination...")
    
    all_sentences = clean_sentences.copy()
    adversarial_map = {}  # sent_id -> {claim_id, type, text}
    claim_to_adv_map = {}  # claim_id -> [sent_ids]
    
    clean_count = len(clean_sentences)
    
    for claim_data in tqdm(adversarial_data.get('claims', []), desc="  Processing claims"):
        claim_id = claim_data.get('event_id')
        
        if 'adversarial_evidence' not in claim_data:
            continue
        
        claim_adv_ids = []
        
        for cont_type, sentence in claim_data['adversarial_evidence'].items():
            if sentence and sentence.strip():
                clean_sent = sentence.strip().strip('"')
                sent_id = len(all_sentences)
                
                all_sentences.append(clean_sent)
                adversarial_map[sent_id] = {
                    'claim_id': claim_id,
                    'type': cont_type,
                    'text': clean_sent
                }
                
                claim_adv_ids.append(sent_id)
        
        if claim_adv_ids:
            claim_to_adv_map[claim_id] = claim_adv_ids
    
    print(f"    ✓ Total sentences: {len(all_sentences)}")
    print(f"      - Clean: {clean_count}")
    print(f"      - Adversarial: {len(adversarial_map)}")
    if len(all_sentences) > 0:
        print(f"      - Contamination rate: {len(adversarial_map)/len(all_sentences)*100:.2f}%")
    
    return all_sentences, adversarial_map, claim_to_adv_map


# ============================================================================
# FAISS INDEX BUILDING (unchanged)
# ============================================================================

def build_faiss_index(sentences: List[str],
                     model_name: str,
                     model_key: str,
                     batch_size: int = BATCH_SIZE) -> Tuple:
    """
    Build FAISS index with progress tracking
    Returns: (index, embeddings, model)
    """
    print(f"  Building FAISS index with {model_key} ({model_name})...")
    print(f"    Encoding {len(sentences)} sentences...")
    
    # Load model
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    # Encode in batches
    all_embeddings = []
    
    for i in tqdm(range(0, len(sentences), batch_size), desc="  Encoding batches"):
        batch = sentences[i:i+batch_size]
        
        batch_embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=16
        )
        
        all_embeddings.append(batch_embeddings)
        
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Combine all batches
    embeddings = np.vstack(all_embeddings)
    print(f"    ✓ Embeddings shape: {embeddings.shape}")
    
    # Build FAISS index
    print("    Building FAISS IndexFlatIP...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    
    print(f"    ✓ Index built with {index.ntotal} vectors")
    
    return index, embeddings, model


# ============================================================================
# SAVE/LOAD FUNCTIONS (unchanged)
# ============================================================================

def save_index_and_metadata(save_dir: str,
                            dataset_name: str,
                            split: str,
                            model_key: str,
                            model_name: str,
                            index: faiss.Index,
                            embeddings: np.ndarray,
                            sentences: List[str],
                            adversarial_map: Dict,
                            claim_to_adv_map: Dict):
    """Save index, embeddings, and metadata"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    base_name = f"{save_dir}/{dataset_name.lower()}_{split}_{model_key}"
    
    print(f"  Saving to {base_name}...")
    
    # Save FAISS index
    faiss.write_index(index, f"{base_name}.index")
    print(f"    ✓ Saved: {base_name}.index")
    
    # Save embeddings
    np.save(f"{base_name}_embeddings.npy", embeddings)
    print(f"    ✓ Saved: {base_name}_embeddings.npy")
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'split': split,
        'sentences': sentences,
        'adversarial_map': adversarial_map,
        'claim_to_adv_map': claim_to_adv_map,
        'model_name': model_name,
        'model_key': model_key,
        'n_clean': len(sentences) - len(adversarial_map),
        'n_adversarial': len(adversarial_map),
        'contamination_rate': len(adversarial_map) / len(sentences) if len(sentences) > 0 else 0,
        'created_at': datetime.now().isoformat()
    }
    
    with open(f"{base_name}_metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    print(f"    ✓ Saved: {base_name}_metadata.pkl")
    
    # Save summary as JSON
    summary = {
        'dataset': dataset_name,
        'split': split,
        'model': model_name,
        'model_key': model_key,
        'total_sentences': len(sentences),
        'clean_sentences': len(sentences) - len(adversarial_map),
        'adversarial_sentences': len(adversarial_map),
        'contamination_rate': f"{len(adversarial_map) / len(sentences) * 100:.2f}%" if len(sentences) > 0 else "0.00%",
        'n_claims_with_adversarial': len(claim_to_adv_map),
        'index_dimension': embeddings.shape[1],
        'created_at': datetime.now().isoformat()
    }
    
    with open(f"{base_name}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"    ✓ Saved: {base_name}_summary.json")


def load_existing_index(save_dir: str,
                       dataset_name: str,
                       split: str,
                       model_key: str,
                       model_name: str) -> Tuple:
    """Load pre-built index and metadata"""
    
    base_name = f"{save_dir}/{dataset_name.lower()}_{split}_{model_key}"
    
    print(f"Loading index from {base_name}...")
    
    # Load FAISS index
    index = faiss.read_index(f"{base_name}.index")
    print(f"  ✓ Loaded FAISS index ({index.ntotal} vectors)")
    
    # Load embeddings
    embeddings = np.load(f"{base_name}_embeddings.npy")
    print(f"  ✓ Loaded embeddings {embeddings.shape}")
    
    # Load metadata
    with open(f"{base_name}_metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    print(f"  ✓ Loaded metadata ({len(metadata['sentences'])} sentences)")
    
    # Load model
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print(f"  ✓ Loaded model")
    
    return index, embeddings, metadata, model


# ============================================================================
# MAIN BUILD PIPELINE
# ============================================================================

def build_all_indices():
    """Build indices for all combinations of datasets, splits, and models"""
    
    print("=" * 90)
    print("COMPLETE FAISS INDEX BUILDER (FIXED)")
    print("=" * 90)
    print(f"Save directory: {SAVE_DIR}")
    print(f"Datasets: {list(DATASET_CONFIGS.keys())}")
    print(f"Models: {list(MODELS.keys())}")
    print("=" * 90)
    print()
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    build_summary = {
        'started_at': datetime.now().isoformat(),
        'datasets': {},
        'total_indices': 0,
        'failed_builds': []
    }
    
    # Iterate through all combinations
    for dataset_name, dataset_config in DATASET_CONFIGS.items():
        print(f"\n{'='*90}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*90}")
        
        build_summary['datasets'][dataset_name] = {}
        
        for split in dataset_config['splits']:
            print(f"\n{'-'*90}")
            print(f"Split: {split}")
            print(f"{'-'*90}")
            
            try:
                # Load dataset
                data = load_dataset_split(
                    dataset_name,
                    dataset_config['base_path'],
                    split
                )
                
                # Extract clean sentences (SAME for both RAWFC and LIAR-RAW!)
                clean_sentences = extract_all_sentences(dataset_name, data)
                
                # Load adversarial data
                adv_path = dataset_config['adversarial_paths'].get(split)
                adversarial_data = load_adversarial_data(adv_path) if adv_path else None
                
                # Add contamination
                all_sentences, adversarial_map, claim_to_adv_map = add_adversarial_contamination(
                    clean_sentences, adversarial_data
                )
                
                # Build index for each model
                for model_key, model_name in MODELS.items():
                    print(f"\n  {'·'*40}")
                    print(f"  Model: {model_key}")
                    print(f"  {'·'*40}")
                    
                    try:
                        # Build FAISS index
                        index, embeddings, model = build_faiss_index(
                            all_sentences, model_name, model_key, BATCH_SIZE
                        )
                        
                        # Save everything
                        save_index_and_metadata(
                            SAVE_DIR, dataset_name, split, model_key, model_name,
                            index, embeddings, all_sentences,
                            adversarial_map, claim_to_adv_map
                        )
                        
                        print(f"  ✅ Completed {dataset_name}/{split}/{model_key}")
                        
                        # Record success
                        key = f"{split}_{model_key}"
                        build_summary['datasets'][dataset_name][key] = {
                            'status': 'success',
                            'total_sentences': len(all_sentences),
                            'clean': len(all_sentences) - len(adversarial_map),
                            'adversarial': len(adversarial_map)
                        }
                        build_summary['total_indices'] += 1
                        
                        # Clear GPU memory
                        del model, index, embeddings
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                    except Exception as e:
                        error_msg = f"{dataset_name}/{split}/{model_key}: {str(e)}"
                        print(f"  ❌ FAILED: {error_msg}")
                        build_summary['failed_builds'].append(error_msg)
                        continue
                
            except Exception as e:
                error_msg = f"{dataset_name}/{split}: {str(e)}"
                print(f"❌ FAILED: {error_msg}")
                build_summary['failed_builds'].append(error_msg)
                continue
    
    # Save build summary
    build_summary['completed_at'] = datetime.now().isoformat()
    summary_path = os.path.join(SAVE_DIR, 'build_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(build_summary, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 90)
    print("BUILD COMPLETE!")
    print("=" * 90)
    print(f"Total indices built: {build_summary['total_indices']}")
    print(f"Failed builds: {len(build_summary['failed_builds'])}")
    
    if build_summary['failed_builds']:
        print("\nFailed builds:")
        for error in build_summary['failed_builds']:
            print(f"   {error}")
    
    print(f"\nAll files saved to: {SAVE_DIR}/")
    print(f"Build summary: {summary_path}")
    print("=" * 90)


def verify_index(dataset_name: str, split: str, model_key: str, test_claim: str = None):
    """Verify that an index works correctly"""
    
    model_name = MODELS[model_key]
    
    print(f"\n{'='*90}")
    print(f"VERIFYING: {dataset_name}/{split}/{model_key}")
    print(f"{'='*90}")
    
    try:
        # Load index
        index, embeddings, metadata, model = load_existing_index(
            SAVE_DIR, dataset_name, split, model_key, model_name
        )
        
        sentences = metadata['sentences']
        adversarial_map = metadata['adversarial_map']
        
        # Print statistics
        print(f"\n Index Statistics:")
        print(f"  Total sentences: {len(sentences)}")
        print(f"  Clean sentences: {len(sentences) - len(adversarial_map)}")
        print(f"  Adversarial sentences: {len(adversarial_map)}")
        if len(sentences) > 0:
            print(f"  Contamination rate: {len(adversarial_map)/len(sentences)*100:.2f}%")
        
        # Verify clean sentences exist
        if len(sentences) - len(adversarial_map) == 0:
            print("\n  ⚠️ WARNING: No clean sentences found!")
            print("  This suggests the sentence extraction failed.")
        else:
            print(f"\n  Clean sentences present: {len(sentences) - len(adversarial_map)}")
        
        # Test query
        if not test_claim:
            test_claims = {
                'RAWFC': "Kamala Harris wants to ban gas-powered cars",
                'LIAR-RAW': "Says Donald Trump wants to cut Social Security"
            }
            test_claim = test_claims.get(dataset_name, "This is a test claim")
        
        print(f"\n Test Query: {test_claim}")
        
        # Encode query
        query_emb = model.encode([test_claim], normalize_embeddings=True, convert_to_numpy=True)
        
        # Search
        k = 20
        distances, indices = index.search(query_emb.astype('float32'), k)
        
        print(f"\nTop-{k} Results:")
        print("-" * 90)
        
        n_adversarial = 0
        n_clean = 0
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
            idx = int(idx)
            is_adv = idx in adversarial_map
            marker = "🚨 ADV" if is_adv else "✅ Clean"
            
            if is_adv:
                n_adversarial += 1
                adv_info = adversarial_map[idx]
                print(f"{i:2d}. {marker} [{adv_info['type']}] (score: {dist:.4f})")
            else:
                n_clean += 1
                print(f"{i:2d}. {marker} (score: {dist:.4f})")
            
            # Truncate long sentences
            sent_preview = sentences[idx][:100]
            if len(sentences[idx]) > 100:
                sent_preview += "..."
            print(f"    {sent_preview}")
        
        print("-" * 90)
        print(f" Retrieved: {n_clean} clean + {n_adversarial} adversarial ({n_adversarial/k*100:.1f}%)")
        
        # Validation
        if len(sentences) - len(adversarial_map) == 0:
            print("\n FAILED: No clean sentences in index!")
            print("   → Need to rebuild with correct sentence extraction")
            return False
        elif n_clean == 0:
            print("\n WARNING: No clean sentences retrieved despite being in index")
            print("   → Adversarial contamination might be dominating")
        else:
            print("\n Index is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n VERIFICATION FAILED: {str(e)}")
        return False


def verify_all_indices():
    """Verify all built indices"""
    
    print("\n" + "=" * 90)
    print("VERIFYING ALL INDICES")
    print("=" * 90)
    
    verification_results = {}
    
    for dataset_name in DATASET_CONFIGS.keys():
        verification_results[dataset_name] = {}
        
        for split in DATASET_CONFIGS[dataset_name]['splits']:
            for model_key in MODELS.keys():
                key = f"{split}_{model_key}"
                
                success = verify_index(dataset_name, split, model_key)
                
                verification_results[dataset_name][key] = success
    
    # Summary
    print("\n" + "=" * 90)
    print("VERIFICATION SUMMARY")
    print("=" * 90)
    
    for dataset_name, results in verification_results.items():
        print(f"\n{dataset_name}:")
        for key, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {key}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "verify":
            verify_all_indices()
            
        elif command == "verify-one":
            if len(sys.argv) < 5:
                print("Usage: python script.py verify-one <dataset> <split> <model>")
                print("Example: python script.py verify-one LIAR-RAW train qwen")
                sys.exit(1)
            
            dataset_name = sys.argv[2]
            split = sys.argv[3]
            model_key = sys.argv[4]
            
            verify_index(dataset_name, split, model_key)
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: verify, verify-one")
            sys.exit(1)
    
    else:
        # Build all indices
        build_all_indices()
        
        # Ask if user wants to verify
        print("\n" + "=" * 90)
        response = input("Run verification on all indices? (y/n): ")
        if response.lower() == 'y':
            verify_all_indices()


if __name__ == "__main__":
    main()
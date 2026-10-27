import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import warnings
import time
from datetime import datetime
import argparse

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



ST_MODEL_NAME = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
BATCH_SIZE = 8
LEARNING_RATE = 5e-6
EPOCH_SIZE = 5
TOP_K = 20
K_FINAL = 10
CACHE_DIR = './cached_indices'

ADVERSARIAL_PATHS = {
    'RAWFC': {
        'train': "./dataset/RAWFC/rawfc-train-gemma3-adversarial.json",
        'val': "./dataset/RAWFC/rawfc-val-gemma3-adversarial.json",
        'test': "./dataset/RAWFC/rawfc-test-gemma3-adversarial.json",
    },
    'LIAR-RAW': {
        'train': "./dataset/LIAR-RAW/liarraw-train-gemma3-adversarial.json",
        'val': "./dataset/LIAR-RAW/liarraw-val-gemma3-adversarial.json",
        'test': "./dataset/LIAR-RAW/liarraw-test-gemma3-adversarial.json",
    }
}

DATASET_CONFIGS = {
    'RAWFC': {
        'num_labels': 3,
        'label_map': {"false": 0, "half": 1, "true": 2},
    },
    'LIAR-RAW': {
        'num_labels': 6,
        'label_map': {
            "pants-fire": 0, "false": 1, "barely-true": 2,
            "half-true": 3, "mostly-true": 4, "true": 5
        },
    }
}

# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------

CONTAM_MODES = [
    'clean',
    'contaminated',
    'trustrag',
    'ragdefender',
    'glad_dense',
     'glad',
     'glad_lexical',
]

CLEAN_DEFENSE_MODES = [
    'trustrag_clean',
    'ragdefender_clean',
    'glad_dense_clean',
    'glad_clean',
     'glad_lexical_clean', 
]

ALL_MODES = CONTAM_MODES + CLEAN_DEFENSE_MODES

SEED = 43  # overwritten by --seed


# ============================================================================
# HELPERS
# ============================================================================

def _uses_clean_cache(mode: str) -> bool:
    """Modes that read from the un-contaminated train index."""
    return mode == 'clean' or mode.endswith('_clean')


def _base_filter_mode(mode: str) -> str:
    """Strip a `_clean` suffix to find the underlying filter."""
    if mode.endswith('_clean') and mode != 'clean':
        return mode.rsplit('_clean', 1)[0]
    return mode


def _normalize(x):
    rng = x.max() - x.min()
    if rng == 0:
        return np.zeros_like(x)
    return (x - x.min()) / rng


def _fix_zero_vectors(tfidf):
    zero_mask = np.all(tfidf == 0, axis=1)
    if zero_mask.any():
        tfidf[zero_mask] = np.random.RandomState(42).uniform(
            1e-10, 1e-8, size=(zero_mask.sum(), tfidf.shape[1]))
    return tfidf


# ============================================================================
# DATASET READER
# ============================================================================

def read_dataset(dataset_name, split):
    base = os.path.join(".", "dataset", dataset_name)
    if dataset_name == 'LIAR-RAW':
        fp = os.path.join(base, f"{split}.json")
        with open(fp, 'r') as f:
            return json.load(f)
    else:
        sp = os.path.join(base, split)
        data = []
        for fn in sorted(os.listdir(sp)):
            if fn.endswith('.json'):
                with open(os.path.join(sp, fn), 'r') as f:
                    d = json.load(f)
                    data.extend(d if isinstance(d, list) else [d])
        return data


class IndexCache:
    def __init__(self, dataset_name, st_model_name, cache_dir=CACHE_DIR):
        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        self.cache_dir = cache_dir
        self.st_model_name = st_model_name
        self._st_model = None
        self.cache = {}
        os.makedirs(cache_dir, exist_ok=True)

    @property
    def st_model(self):
        if self._st_model is None:
            print(f"Loading sentence transformer: {self.st_model_name}")
            self._st_model = SentenceTransformer(self.st_model_name, trust_remote_code=True)
        return self._st_model

    def _cache_key(self, split, contaminate):
        model_key = 'qwen' if 'qwen' in self.st_model_name.lower() else 'nomic'
        contam_tag = 'contam' if contaminate else 'clean'
        return f"{self.dataset_name}_{model_key}_{split}_{contam_tag}_k{TOP_K}"

    def _cache_path(self, cache_key):
        return os.path.join(self.cache_dir, f"{cache_key}.npz")

    def _meta_path(self, cache_key):
        return os.path.join(self.cache_dir, f"{cache_key}_meta.json")

    def build_and_cache(self, dataset, split, contaminate=False):
        cache_key = self._cache_key(split, contaminate)
        cache_file = self._cache_path(cache_key)
        meta_file = self._meta_path(cache_key)

        # --- Try loading from disk ---
        if os.path.exists(cache_file) and os.path.exists(meta_file):
            print(f"\n{'='*70}")
            print(f"LOADING CACHED INDEX: {cache_key}")
            print(f"{'='*70}")
            t0 = time.time()
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            data = np.load(cache_file, allow_pickle=True)

            retrievals = []
            n_claims = meta['n_claims']
            for i in range(n_claims):
                retrievals.append({
                    'claim': meta['claims'][i],
                    'sentences': meta['retrieval_sentences'][i],
                    'embeddings': data[f'embs_{i}'],
                    'claim_emb': data[f'claim_emb_{i}'],
                    'adv_flags': meta['retrieval_adv_flags'][i],
                })

            self.cache[split] = {
                'claims': meta['claims'],
                'labels': meta['labels'],
                'retrievals': retrievals,
                'adversarial_map': {int(k): v for k, v in meta.get('adversarial_map', {}).items()},
            }
            total_adv = sum(sum(r['adv_flags']) for r in retrievals)
            claims_with_adv = sum(1 for r in retrievals if any(r['adv_flags']))
            print(f"  Loaded {n_claims} cached retrievals in {time.time()-t0:.1f}s")
            print(f"  Total adversarial in top-{TOP_K}: {total_adv}")
            print(f"  Claims with >=1 adversarial: {claims_with_adv}")
            return time.time() - t0

        # --- Build from scratch ---
        print(f"\n{'='*70}")
        print(f"BUILDING INDEX + CACHING RETRIEVALS: {split.upper()}")
        print(f"  Contaminate: {contaminate}")
        print(f"{'='*70}")
        t0 = time.time()

        all_sentences = []
        for item in tqdm(dataset, desc="Extracting sentences"):
            if 'reports' in item:
                for report in item['reports']:
                    if 'tokenized' in report:
                        for s in report['tokenized']:
                            if isinstance(s, dict) and 'sent' in s:
                                all_sentences.append(s['sent'])

        print(f"  Clean sentences: {len(all_sentences)}")

        adversarial_map = {}
        if contaminate:
            adv_paths = ADVERSARIAL_PATHS.get(self.dataset_name, {})
            adv_file = adv_paths.get(split if split != 'train_clean' else 'train')

            if adv_file and os.path.exists(adv_file):
                with open(adv_file, 'r') as f:
                    adv_data = json.load(f)
                print(f"  Loaded adversarial file: {adv_file}")

                for claim_data in adv_data.get('claims', []):
                    if 'adversarial_evidence' not in claim_data:
                        continue
                    for cont_type, sentence in claim_data['adversarial_evidence'].items():
                        if sentence and sentence.strip():
                            sent_id = len(all_sentences)
                            all_sentences.append(sentence.strip().strip('"'))
                            adversarial_map[sent_id] = {
                                'event_id': claim_data.get('event_id'),
                                'type': cont_type,
                            }
                print(f"  Adversarial injected: {len(adversarial_map)}")
                print(f"  Total: {len(all_sentences)} "
                      f"({len(adversarial_map)/len(all_sentences)*100:.2f}% contamination)")
            else:
                msg = adv_file or "(not configured)"
                print(f"  Adversarial file not found: {msg}")

        print(f"  Encoding {len(all_sentences)} sentences...")
        chunk_size = 5000
        all_embs = []
        for i in range(0, len(all_sentences), chunk_size):
            chunk = all_sentences[i:i + chunk_size]
            emb = self.st_model.encode(chunk, show_progress_bar=True,
                                        normalize_embeddings=True, batch_size=16)
            if torch.is_tensor(emb):
                emb = emb.cpu().numpy()
            all_embs.append(emb)
            torch.cuda.empty_cache()
        embeddings = np.vstack(all_embs)
        del all_embs

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype('float32'))
        print(f"  FAISS index: {index.ntotal} vectors, dim={dim}")

        claims, labels, claim_texts = [], [], []
        for item in dataset:
            if self.dataset_name == 'LIAR-RAW':
                claim = item.get('statement') or item.get('claim')
            else:
                claim = item.get('claim')
            label = item.get('label')
            if not claim or not label:
                continue
            lk = label.lower()
            if lk not in self.config['label_map']:
                continue
            claim_texts.append(claim)
            labels.append(self.config['label_map'][lk])

        print(f"  Retrieving top-{TOP_K} for {len(claim_texts)} claims...")
        all_retrievals = []
        for claim in tqdm(claim_texts, desc="Retrieving"):
            claim_emb = self.st_model.encode([claim], normalize_embeddings=True)
            if torch.is_tensor(claim_emb):
                claim_emb = claim_emb.cpu().numpy()
            distances, indices = index.search(claim_emb.astype('float32'), TOP_K)

            ret_sents, ret_embs, ret_adv = [], [], []
            for idx in indices[0]:
                idx = int(idx)
                ret_sents.append(all_sentences[idx])
                ret_embs.append(embeddings[idx])
                ret_adv.append(idx in adversarial_map)

            all_retrievals.append({
                'claim': claim,
                'sentences': ret_sents,
                'embeddings': np.array(ret_embs),
                'claim_emb': claim_emb,
                'adv_flags': ret_adv,
            })

        total_adv = sum(sum(r['adv_flags']) for r in all_retrievals)
        claims_with_adv = sum(1 for r in all_retrievals if any(r['adv_flags']))
        build_time = time.time() - t0

        print(f"\n  Cached {len(all_retrievals)} retrievals in {build_time:.1f}s")
        print(f"  Total adversarial in top-{TOP_K}: {total_adv}")
        print(f"  Claims with >=1 adversarial: {claims_with_adv}")

        self.cache[split] = {
            'claims': claim_texts,
            'labels': labels,
            'retrievals': all_retrievals,
            'adversarial_map': adversarial_map,
        }

        # --- Save to disk ---
        print(f"  Saving cache to disk: {cache_key}")
        t_save = time.time()
        np_data = {}
        for i, ret in enumerate(all_retrievals):
            np_data[f'embs_{i}'] = ret['embeddings']
            np_data[f'claim_emb_{i}'] = ret['claim_emb']
        np.savez_compressed(cache_file, **np_data)

        meta = {
            'n_claims': len(all_retrievals),
            'claims': claim_texts,
            'labels': labels,
            'adversarial_map': {str(k): v for k, v in adversarial_map.items()},
            'retrieval_sentences': [r['sentences'] for r in all_retrievals],
            'retrieval_adv_flags': [r['adv_flags'] for r in all_retrievals],
            'total_adv_in_topk': total_adv,
            'model': self.st_model_name,
            'top_k': TOP_K,
        }
        with open(meta_file, 'w') as f:
            json.dump(meta, f)

        print(f"  Saved in {time.time()-t_save:.1f}s "
              f"({os.path.getsize(cache_file)/1024/1024:.1f}MB npz + "
              f"{os.path.getsize(meta_file)/1024/1024:.1f}MB meta)")
        return build_time



def retrieval_penetration_analysis(cache: IndexCache):
    train_cache = cache.cache.get('train')
    if train_cache is None:
        print("No contaminated train cache")
        return None

    retrievals = train_cache['retrievals']
    k_values = [3, 5, 10]
    print(f"\n{'='*80}")
    print(f" RETRIEVAL PENETRATION ANALYSIS")
    print(f"{'='*80}")

    results = {}
    for k in k_values:
        total_claims = len(retrievals)
        claims_with_adv = 0
        total_adv_in_topk = 0
        for ret in retrievals:
            adv_flags_k = ret['adv_flags'][:k]
            n_adv = sum(adv_flags_k)
            total_adv_in_topk += n_adv
            if n_adv > 0:
                claims_with_adv += 1
        rpr = claims_with_adv / total_claims if total_claims > 0 else 0
        avg_adv_pct = (total_adv_in_topk / (total_claims * k) * 100) if total_claims > 0 else 0
        results[k] = {
            'total_claims': total_claims,
            'claims_with_adv': claims_with_adv,
            'rpr': rpr,
            'rpr_pct': rpr * 100,
            'total_adv_in_topk': total_adv_in_topk,
            'avg_adv_per_claim': total_adv_in_topk / total_claims if total_claims > 0 else 0,
            'avg_adv_pct_of_k': avg_adv_pct,
        }
        print(f"\n  K={k}:")
        print(f"    RPR: {claims_with_adv}/{total_claims} ({rpr*100:.1f}%)")
        print(f"    Total adversarial in top-{k}: {total_adv_in_topk}")
        print(f"    Adversarial as % of top-{k}: {avg_adv_pct:.1f}%")
    return results


def _filter_trustrag(embs, k_final):
    """TrustRAG: K-means in embedding space, drop minority cluster."""
    n = len(embs)
    if n <= k_final:
        return list(range(n))
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = km.fit_predict(embs)
    sizes = np.bincount(labels, minlength=2)
    remove_label = int(np.argmin(sizes))
    kept = np.where(labels != remove_label)[0]
    if len(kept) > k_final:
        keep_label = 1 - remove_label
        centroid = km.cluster_centers_[keep_label]
        dists = np.linalg.norm(embs[kept] - centroid, axis=1)
        kept = kept[np.argsort(dists)[:k_final]]
    elif len(kept) < k_final:
        removed = np.where(labels == remove_label)[0]
        centroid = km.cluster_centers_[1 - remove_label]
        dists = np.linalg.norm(embs[removed] - centroid, axis=1)
        need = k_final - len(kept)
        extra = removed[np.argsort(dists)[:need]]
        kept = np.concatenate([kept, extra])
    return sorted(kept.tolist())


def _filter_ragdefender(sents, embs, k_final):
    """
    RAGDefender 
    Stage 1 (TF-IDF top-term clustering): unigrams, 2 clusters, top m=5 terms.
      - Eq. 1: count passages containing > m/2 of top terms.
      - Eq. 2: estimate N_adv from cluster size + top-term count.

    Stage 2 (similarity pair-frequency): identify adversarial via
    frequency of appearing in the highest-similarity pairs (Eq. 4–7).
    """
    n = len(sents)
    if n <= k_final:
        return list(range(n))

    # Stage 1: estimate N_adv
    n_clusters = min(2, n)
    if n_clusters < 2:
        return list(range(min(n, k_final)))
    try:
        vec = TfidfVectorizer(max_features=5000, stop_words='english',
                              ngram_range=(1, 1), min_df=1)
        tfidf = vec.fit_transform(sents).toarray()
    except ValueError:
        return list(range(min(n, k_final)))
    tfidf = _fix_zero_vectors(tfidf)

    try:
        clust = AgglomerativeClustering(
            n_clusters=n_clusters, metric='cosine', linkage='average')
        tfidf_labels = clust.fit_predict(tfidf)
    except ValueError:
        return list(range(min(n, k_final)))

    tfidf_scores = tfidf.sum(axis=0)
    m = min(5, tfidf.shape[1])
    top_term_indices = np.argsort(-tfidf_scores)[:m]
    has_top = (tfidf[:, top_term_indices] > 0).sum(axis=1) > m / 2
    n_tfidf = int(has_top.sum())

    sizes = np.bincount(tfidf_labels, minlength=n_clusters)
    n_min = int(sizes.min())
    if n_tfidf <= n // 2:
        n_adv = n_min
    else:
        n_adv = n - n_min
    n_adv = max(1, min(n_adv, n - k_final))

    # Stage 2: pair-frequency scoring
    sim = cosine_similarity(embs)
    np.fill_diagonal(sim, -1)
    n_pairs = max(1, n_adv // 2)

    flat_idx = np.argsort(sim.ravel())[::-1]
    freq_scores = np.zeros(n)
    pairs_found = 0
    seen = set()
    for idx in flat_idx:
        i, j = divmod(idx, n)
        if i >= j:
            continue
        if (i, j) in seen:
            continue
        seen.add((i, j))
        s = sim[i, j]
        weighted = np.sign(s) * abs(s) ** 2  # p=2
        freq_scores[i] += weighted
        freq_scores[j] += weighted
        pairs_found += 1
        if pairs_found >= n_pairs:
            break

    adv_indices = set(np.argsort(-freq_scores)[:n_adv].tolist())
    kept = [i for i in range(n) if i not in adv_indices]
    if len(kept) > k_final:
        kept_scores = [(i, freq_scores[i]) for i in kept]
        kept_scores.sort(key=lambda x: x[1])
        kept = [i for i, _ in kept_scores[:k_final]]
    return sorted(kept)


# ----------------------------------------------------------------------------
# GLAD components
# ----------------------------------------------------------------------------

def _glad_dense_scores(embs):
    """
    GLAD dense signal (formerly "SCOD").

    Geometric peer-coherence in dense embedding space.
    Higher score = more suspicious (lower coherence with peers).
    """
    n = len(embs)
    if n < 2:
        return np.zeros(n)
    sim = cosine_similarity(embs)
    coherence = np.array([
        np.mean(np.concatenate([sim[i, :i], sim[i, i+1:]]))
        for i in range(n)
    ])
    return 1 - coherence


def _glad_sparse_scores(sents):
    """
    GLAD sparse signal (formerly the `_ragdefender_scores` helper that
    only existed inside the GLAD ensemble).

    Distributional isolation in sparse TF-IDF space:
      0.6 * cosine_dist(sentence, main-cluster centroid)
      + 0.4 * (1 - |cluster(sentence)| / k)
    Higher score = more isolated / suspicious.

    NOTE: This is NOT the paper-faithful Kim et al. RAGDefender. It is a
    simpler distributional method (bigrams, 3 clusters, centroid-distance
    + size-penalty), used as the sparse signal of GLAD. The standalone
    `_filter_ragdefender` above is the paper-faithful version.
    """
    n = len(sents)
    if n < 3:
        return np.zeros(n)
    try:
        vec = TfidfVectorizer(max_features=5000, stop_words='english',
                              ngram_range=(1, 2), min_df=1)
        tfidf = vec.fit_transform(sents).toarray()
    except ValueError:
        return np.zeros(n)
    tfidf = _fix_zero_vectors(tfidf)
    nc = min(3, n)
    try:
        clust = AgglomerativeClustering(
            n_clusters=nc, metric='cosine', linkage='average')
        labels = clust.fit_predict(tfidf)
    except ValueError:
        return np.zeros(n)
    sizes = np.bincount(labels, minlength=nc)
    largest = np.argmax(sizes)
    main_centroid = tfidf[labels == largest].mean(axis=0, keepdims=True)
    distances = cosine_distances(tfidf, main_centroid).flatten()
    penalty = np.array([1.0 - (sizes[labels[i]] / n) for i in range(n)])
    return 0.6 * distances + 0.4 * penalty


def _keep_least_suspicious(scores, k_final):
    """Return indices of the k_final LEAST suspicious sentences."""
    return np.argsort(scores)[:k_final].tolist()


def _filter_glad_dense(embs, k_final):
    n = len(embs)
    if n <= k_final:
        return list(range(n))
    return _keep_least_suspicious(_glad_dense_scores(embs), k_final)


def _filter_glad_lexical(sents, k_final):
    n = len(sents)
    if n <= k_final:
        return list(range(n))
    return _keep_least_suspicious(_glad_sparse_scores(sents), k_final)

def _filter_glad(sents, embs, k_final):
    """GLAD = 0.5 * normalized(glad_dense) + 0.5 * normalized(glad_sparse)."""
    n = len(sents)
    if n <= k_final:
        return list(range(n))
    s1 = _normalize(_glad_dense_scores(embs))
    s2 = _normalize(_glad_sparse_scores(sents))
    return _keep_least_suspicious(0.5 * s1 + 0.5 * s2, k_final)


# ============================================================================
# APPLY FILTER (unified dispatch)
# ============================================================================

FILTER_DISPATCH = {
    'trustrag':    lambda sents, embs: _filter_trustrag(embs, K_FINAL),
    'ragdefender': lambda sents, embs: _filter_ragdefender(sents, embs, K_FINAL),
    'glad_dense':  lambda sents, embs: _filter_glad_dense(embs, K_FINAL),
    'glad_lexical': lambda sents, embs: _filter_glad_lexical(sents, K_FINAL),
    'glad':        lambda sents, embs: _filter_glad(sents, embs, K_FINAL),

}


def apply_filter(mode, cached_retrievals):
    filter_mode = _base_filter_mode(mode)

    evidence_texts = []
    total_adv_before = 0
    total_adv_after = 0

    for ret in cached_retrievals:
        sents = ret['sentences']
        embs = ret['embeddings']
        adv_flags = ret['adv_flags']
        n = len(sents)

        total_adv_before += sum(adv_flags)

        if n <= K_FINAL:
            keep = list(range(n))
        elif filter_mode in ('clean', 'contaminated'):
            keep = list(range(min(n, K_FINAL)))
        elif filter_mode in FILTER_DISPATCH:
            keep = FILTER_DISPATCH[filter_mode](sents, embs)
        else:
            raise ValueError(f"Unknown filter mode: {filter_mode} (from mode={mode})")

        filtered_sents = [sents[i] for i in keep]
        filtered_adv = [adv_flags[i] for i in keep]
        total_adv_after += sum(filtered_adv)
        evidence_texts.append(' '.join(filtered_sents))

    stats = {
        'adv_before': total_adv_before,
        'adv_after': total_adv_after,
        'adv_filtered': total_adv_before - total_adv_after,
        'filter_rate': ((total_adv_before - total_adv_after) / total_adv_before * 100
                        if total_adv_before > 0 else 0),
    }
    return evidence_texts, stats

class SimpleDataset(Dataset):
    def __init__(self, claims, evidences, labels, tokenizer, max_length=512):
        self.claims = claims
        self.evidences = evidences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        text = f"claim: {self.claims[idx]} [SEP] evidence: {self.evidences[idx]}"
        enc = self.tokenizer(text, max_length=self.max_length,
                              padding='max_length', truncation=True,
                              return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_and_evaluate(dataset_name,
                       train_claims, train_evidence, train_labels,
                       val_claims, val_evidence, val_labels,
                       test_claims, test_evidence, test_labels,
                       save_dir, seed):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = DATASET_CONFIGS[dataset_name]

    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, add_prefix_space=True, use_fast=True)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[SEP]']})
    classifier = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=config['num_labels']).to(device)
    classifier.resize_token_embeddings(len(tokenizer))

    train_ds = SimpleDataset(train_claims, train_evidence, train_labels, tokenizer)
    val_ds = SimpleDataset(val_claims, val_evidence, val_labels, tokenizer)
    test_ds = SimpleDataset(test_claims, test_evidence, test_labels, tokenizer)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * EPOCH_SIZE)

    best_f1 = 0
    best_epoch = 0

    for epoch in range(EPOCH_SIZE):
        classifier.train()
        for batch in tqdm(train_loader, desc=f'  Epoch {epoch+1}/{EPOCH_SIZE}', ncols=80):
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = classifier(**inputs)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()

        classifier.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                preds = torch.argmax(classifier(**inputs).logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(inputs['labels'].cpu().numpy())
        val_f1 = f1_score(val_labels_list, val_preds, average='macro', zero_division=0)
        print(f"    Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            os.makedirs(save_dir, exist_ok=True)
            classifier.save_pretrained(os.path.join(save_dir, 'best_model'))
            tokenizer.save_pretrained(os.path.join(save_dir, 'best_model'))

    best_path = os.path.join(save_dir, 'best_model')
    if os.path.exists(best_path):
        classifier = AutoModelForSequenceClassification.from_pretrained(
            best_path, local_files_only=True).to(device)

    classifier.eval()
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            preds = torch.argmax(classifier(**inputs).logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(inputs['labels'].cpu().numpy())

    test_metrics = {
        'accuracy': accuracy_score(test_labels_list, test_preds),
        'precision': precision_score(test_labels_list, test_preds,
                                     average='macro', zero_division=0),
        'recall': recall_score(test_labels_list, test_preds,
                               average='macro', zero_division=0),
        'f1': f1_score(test_labels_list, test_preds,
                       average='macro', zero_division=0),
    }
    return test_metrics, best_epoch


def main():
    parser = argparse.ArgumentParser(description='Cleaned experiment runner')
    parser.add_argument('--dataset', default='RAWFC', choices=['RAWFC', 'LIAR-RAW'])
    parser.add_argument('--model', default=None, help='nomic or qwen')
    parser.add_argument('--modes', nargs='+', default=None,
                        help='Defense modes to run. Default: all CONTAM_MODES.')
    parser.add_argument('--include-clean-defense', action='store_true',
                        help='Also run defenses on the CLEAN train index.')
    parser.add_argument('--retrieval-analysis', action='store_true')
    parser.add_argument('--detection-only', action='store_true',
                        help='Skip classifier training, only compute filtering stats.')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--cache-dir', default=CACHE_DIR)

    args = parser.parse_args()

    dataset_name = args.dataset
    global ST_MODEL_NAME, SEED
    SEED = args.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.model == 'nomic':
        ST_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1.5'
    elif args.model == 'qwen':
        ST_MODEL_NAME = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
    elif args.model is not None:
        ST_MODEL_NAME = args.model

    if args.modes:
        modes = args.modes
    elif args.detection_only:
        # Only modes that do filtering — skip 'clean' (no filtering)
        modes = [m for m in CONTAM_MODES if m != 'clean']
    else:
        modes = CONTAM_MODES.copy()
        if args.include_clean_defense:
            modes.extend(CLEAN_DEFENSE_MODES)

    model_key = 'qwen' if 'qwen' in ST_MODEL_NAME.lower() else 'nomic'

    print(f"\n{'='*80}")
    print(f" CLEANED EXPERIMENT RUNNER ".center(80, '='))
    print(f"{'='*80}")
    print(f"Dataset:            {dataset_name}")
    print(f"Retriever:          {ST_MODEL_NAME} ({model_key})")
    print(f"Detection only:     {args.detection_only}")
    print(f"Retrieval analysis: {args.retrieval_analysis}")
    print(f"Modes:              {modes}")
    print(f"K:                  {TOP_K} -> {K_FINAL}")
    print(f"Epochs:             {EPOCH_SIZE}")
    print(f"Seed:               {SEED}")
    print(f"Cache dir:          {args.cache_dir}")
    print(f"{'='*80}")

    total_start = time.time()

    print("\nLoading datasets...")
    train_data = read_dataset(dataset_name, 'train')
    val_data = read_dataset(dataset_name, 'val')
    test_data = read_dataset(dataset_name, 'test')
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    adv_paths = ADVERSARIAL_PATHS.get(dataset_name, {})
    print(f"\n  Adversarial file check:")
    for split_name, path in adv_paths.items():
        exists = os.path.exists(path)
        print(f"  {split_name}: {'OK' if exists else 'NOT FOUND'} -> {path}")

    needs_contam_train = (
        args.retrieval_analysis or
        any(not _uses_clean_cache(m) and m != 'clean' for m in modes)
    )
    needs_clean_train = any(_uses_clean_cache(m) for m in modes)

    cache = IndexCache(dataset_name, ST_MODEL_NAME, cache_dir=args.cache_dir)

    if needs_contam_train:
        cache.build_and_cache(train_data, 'train', contaminate=True)
        train_adv_count = sum(
            sum(r['adv_flags']) for r in cache.cache['train']['retrievals'])
        if train_adv_count == 0:
            print("\nABORT: 0 adversarial in retrievals")
            return
        print(f"  Contamination verified: {train_adv_count} adversarial in retrievals")

    if needs_clean_train:
        cache.build_and_cache(train_data, 'train_clean', contaminate=False)

    if modes and not args.detection_only:
        cache.build_and_cache(val_data, 'val', contaminate=False)
        cache.build_and_cache(test_data, 'test', contaminate=False)

    retrieval_results = None
    if args.retrieval_analysis:
        retrieval_results = retrieval_penetration_analysis(cache)

    if not modes:
        total_time = time.time() - total_start
        print(f"\nTotal time: {total_time/60:.1f} minutes")
        return

    all_results = []

    for mode in modes:
        print(f"\n\n{'#'*80}")
        print(f"  MODE: {mode}")
        print(f"{'#'*80}")
        mode_start = time.time()

        if _uses_clean_cache(mode):
            train_cache = cache.cache['train_clean']
            cache_label = 'CLEAN'
        else:
            train_cache = cache.cache['train']
            cache_label = 'CONTAMINATED'
        print(f"  Using {cache_label} train cache")

        print(f"  Filtering train evidence ({mode})...")
        train_evidence, train_stats = apply_filter(mode, train_cache['retrievals'])
        print(f"    Adv before: {train_stats['adv_before']}, "
              f"after: {train_stats['adv_after']}, "
              f"filtered: {train_stats['adv_filtered']} "
              f"({train_stats['filter_rate']:.1f}%)")

        mode_time = time.time() - mode_start

        if args.detection_only:
            result = {
                'mode': mode,
                'dataset': dataset_name,
                'model': model_key,
                'train_cache': cache_label,
                'filtering_stats': train_stats,
                'time_seconds': round(mode_time, 1),
            }
            all_results.append(result)
            print(f"\n  {mode}: Detection={train_stats['filter_rate']:.1f}%, "
                  f"Time={mode_time:.1f}s")
            continue

        val_cache = cache.cache['val']
        test_cache = cache.cache['test']
        val_evidence, _ = apply_filter('clean', val_cache['retrievals'])
        test_evidence, _ = apply_filter('clean', test_cache['retrievals'])

        save_dir = (f"./saved_models/{dataset_name.lower()}_{model_key}_{mode}"
                    f"_k{TOP_K}_{K_FINAL}_s{SEED}")
        print(f"  Training classifier...")
        test_metrics, best_epoch = train_and_evaluate(
            dataset_name,
            train_cache['claims'], train_evidence, train_cache['labels'],
            val_cache['claims'], val_evidence, val_cache['labels'],
            test_cache['claims'], test_evidence, test_cache['labels'],
            save_dir, SEED,
        )

        mode_time = time.time() - mode_start
        result = {
            'mode': mode,
            'dataset': dataset_name,
            'model': model_key,
            'seed': SEED,
            'train_cache': cache_label,
            'best_epoch': best_epoch,
            'test_metrics': test_metrics,
            'filtering_stats': train_stats,
            'time_seconds': round(mode_time, 1),
        }
        all_results.append(result)

        print(f"\n  {mode}: F1={test_metrics['f1']:.4f}, "
              f"Acc={test_metrics['accuracy']:.4f}, "
              f"Time={mode_time/60:.1f}m")

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(result, f, indent=2)

    # --- Final summary ---
    total_time = time.time() - total_start

    if args.detection_only:
        print(f"\n\n{'='*90}")
        print(f" DETECTION-ONLY RESULTS ".center(90, '='))
        print(f"{'='*90}")
        print(f"\n{'Mode':<22} {'Cache':<14} {'Adv Before':>10} {'Adv After':>10} "
              f"{'Filtered':>10} {'Detection%':>11}")
        print("-" * 90)
        for r in all_results:
            s = r['filtering_stats']
            print(f"{r['mode']:<22} {r['train_cache']:<14} {s['adv_before']:>10} "
                  f"{s['adv_after']:>10} {s['adv_filtered']:>10} "
                  f"{s['filter_rate']:>10.1f}%")
    else:
        print(f"\n\n{'='*110}")
        print(f" FINAL COMPARISON TABLE ".center(110, '='))
        print(f"{'='*110}")
        print(f"\n{'Mode':<22} {'Cache':<14} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} "
              f"{'AdvFilt%':>9} {'Time':>8}")
        print("-" * 100)
        for r in all_results:
            m = r['test_metrics']
            filt = f"{r['filtering_stats']['filter_rate']:.1f}%"
            dur = f"{r['time_seconds']/60:.1f}m"
            print(f"{r['mode']:<22} {r['train_cache']:<14} {m['f1']:>8.4f} "
                  f"{m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} "
                  f"{filt:>9} {dur:>8}")

        clean_f1 = next((r['test_metrics']['f1'] for r in all_results
                         if r['mode'] == 'clean'), None)
        contam_f1 = next((r['test_metrics']['f1'] for r in all_results
                          if r['mode'] == 'contaminated'), None)

        if clean_f1 is not None and contam_f1 is not None and clean_f1 != contam_f1:
            print(f"\n{'Mode':<22} {'F1':>8} {'Detection%':>11} {'Recovery%':>11}")
            print("-" * 60)
            for r in all_results:
                if r['mode'] in ('clean', 'contaminated'):
                    continue
                if r['train_cache'] != 'CONTAMINATED':
                    continue
                f1 = r['test_metrics']['f1']
                det = r['filtering_stats']['filter_rate']
                recovery = ((f1 - contam_f1) / (clean_f1 - contam_f1) * 100
                            if clean_f1 != contam_f1 else 0)
                print(f"{r['mode']:<22} {f1:>8.4f} {det:>10.1f}% "
                      f"{recovery:>10.1f}%")

    print(f"\nTotal time: {total_time/60:.1f} minutes")

    combined_file = (f"results_clean_{dataset_name.lower()}_{model_key}_s{SEED}_"
                     f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    combined_output = {
        'config': {
            'seed': SEED,
            'top_k': TOP_K,
            'k_final': K_FINAL,
            'epochs': EPOCH_SIZE,
            'model': ST_MODEL_NAME,
            'detection_only': args.detection_only,
        },
        'all_results': all_results,
    }
    if retrieval_results:
        combined_output['retrieval_analysis'] = {
            str(k): v for k, v in retrieval_results.items()
        }
    with open(combined_file, 'w') as f:
        json.dump(combined_output, f, indent=2)
    print(f"Saved to: {combined_file}")


if __name__ == "__main__":
    main()
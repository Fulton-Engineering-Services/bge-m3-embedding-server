#!/usr/bin/env python3
"""
Generate equivalence test fixtures for bge-m3-embedding-server.

Uses the HuggingFace Transformers + ONNX Runtime Python stack to produce
reference dense and sparse embeddings from the BAAI/bge-m3 model at several
sequence lengths. The Rust server's embed pipeline is validated against these
fixtures in tests/equivalence.rs.

Usage:
    pip install transformers torch numpy safetensors
    python scripts/generate_equivalence_fixtures.py

    # Optional overrides:
    BGE_M3_CACHE_DIR=/tmp/bge-m3-cache   # where to cache model files
    FIXTURE_SEQ_LENGTHS=256,512,2048      # comma-separated seq lengths to emit
    FIXTURE_TEXTS_PER_LENGTH=10           # how many texts per fixture length

Output:
    tests/fixtures/equivalence/
        README.md
        manifest.json
        texts_seq_0256.json
        reference_dense_seq_0256.npy
        reference_sparse_seq_0256.json
        ... (one set per seq length)

The generator is run once and the fixtures are committed as binary artifacts.
Re-run when:
  - REPO_REVISION in src/embedder.rs is bumped.
  - seq lengths to test are expanded.
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ID = "BAAI/bge-m3"
# Must match REPO_REVISION in src/embedder.rs exactly.
REPO_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CORPUS_PATH = PROJECT_ROOT / "benches" / "fixtures" / "corpus.json"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "equivalence"

SEQ_LENGTHS = [
    int(x)
    for x in os.environ.get("FIXTURE_SEQ_LENGTHS", "256,512,1024,2048,4096,8192").split(",")
]
TEXTS_PER_LENGTH = int(os.environ.get("FIXTURE_TEXTS_PER_LENGTH", "10"))
CACHE_DIR = os.environ.get("BGE_M3_CACHE_DIR", "/tmp/bge-m3-cache")

# ---------------------------------------------------------------------------
# Sparse linear weights (from the bundled safetensors)
# ---------------------------------------------------------------------------

def load_sparse_weights() -> tuple[np.ndarray, float]:
    """Load the sparse_linear weight and bias from the bundled safetensors file."""
    from safetensors.numpy import load_file

    weights_path = PROJECT_ROOT / "src" / "weights" / "sparse_linear.safetensors"
    tensors = load_file(str(weights_path))
    weight = tensors["weight"].astype(np.float32)  # [1024]
    bias = float(tensors["bias"].astype(np.float32).flat[0])
    assert weight.shape == (1024,), f"Expected weight shape (1024,), got {weight.shape}"
    return weight, bias


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus() -> list[str]:
    """Load texts from the curated benchmark corpus."""
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    texts: list[str] = []
    for scenario in corpus["scenarios"].values():
        texts.extend(scenario["texts"])
    return texts


def select_texts(corpus: list[str], target_seq: int, count: int) -> list[str]:
    """Select `count` texts from corpus that will approximately hit `target_seq` tokens.

    Strategy: for each target_seq, prefer texts that are naturally long enough
    to actually exercise that length. For shorter target_seq values, we truncate
    the texts after tokenization. For longer targets, we repeat/concatenate short texts.
    """
    selected = []
    for i in range(count):
        base = corpus[i % len(corpus)]
        # For long sequences, repeat the text to ensure enough tokens.
        # The tokenizer will truncate to target_seq.
        reps = max(1, (target_seq * 5) // (len(base) + 1))  # rough chars-per-token ≈ 5
        repeated = (base + " ") * reps
        selected.append(repeated)
    return selected


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

def compute_embeddings(
    texts: list[str],
    seq_length: int,
    cache_dir: str,
) -> tuple[np.ndarray, list[dict]]:
    """Compute dense and sparse embeddings for `texts` at `seq_length`.

    Returns:
        dense:  np.ndarray of shape [n, 1024], L2-normalized
        sparse: list of {indices: list[int], values: list[float]}
    """
    from transformers import AutoTokenizer

    print(f"  Loading tokenizer (seq_length={seq_length})...")
    tokenizer = AutoTokenizer.from_pretrained(
        REPO_ID,
        revision=REPO_REVISION,
        cache_dir=cache_dir,
        model_max_length=seq_length,
    )

    print(f"  Tokenizing {len(texts)} texts...")
    encoded = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=seq_length,
        return_tensors="np",
    )
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)

    # Run ONNX inference.
    import onnxruntime as ort
    from huggingface_hub import snapshot_download

    print(f"  Downloading FP32 model (revision={REPO_REVISION[:8]})...")
    model_dir = snapshot_download(
        repo_id=REPO_ID,
        revision=REPO_REVISION,
        cache_dir=cache_dir,
        allow_patterns=["onnx/model.onnx", "onnx/model.onnx_data", "onnx/Constant_7_attr__value"],
    )
    model_path = os.path.join(model_dir, "onnx", "model.onnx")

    print(f"  Running ONNX inference (shape={input_ids.shape})...")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    outputs = sess.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )

    # FP32 BAAI model outputs: sentence_embedding [n, 1024] and token_embeddings [n, seq, 1024].
    output_names = [o.name for o in sess.get_outputs()]
    out_map = dict(zip(output_names, outputs))

    # Dense: L2-normalize sentence_embedding.
    sentence_emb = out_map["sentence_embedding"].astype(np.float32)  # [n, 1024]
    norms = np.linalg.norm(sentence_emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    dense = sentence_emb / norms  # [n, 1024], L2-normalized

    # Sparse: token_embeddings → sparse_linear → ReLU → max-pool by token id.
    token_emb = out_map["token_embeddings"].astype(np.float32)  # [n, seq, 1024]
    weight, bias = load_sparse_weights()

    SPECIAL_TOKENS = {0, 1, 2, 3}

    sparse: list[dict] = []
    for i in range(len(texts)):
        ids = input_ids[i]
        mask = attention_mask[i]
        hidden = token_emb[i]  # [seq, 1024]

        scores = hidden @ weight + bias  # [seq]
        scores = np.maximum(scores, 0.0)  # ReLU

        token_weights: dict[int, float] = {}
        for j, (token_id, m, score) in enumerate(zip(ids, mask, scores)):
            if m == 0:
                continue
            tid = int(token_id)
            if tid in SPECIAL_TOKENS:
                continue
            if score > 0.0:
                if tid not in token_weights or score > token_weights[tid]:
                    token_weights[tid] = float(score)

        sorted_ids = sorted(token_weights.keys())
        sparse.append({
            "indices": sorted_ids,
            "values": [token_weights[k] for k in sorted_ids],
        })

    return dense, sparse


# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"BAAI/bge-m3 revision: {REPO_REVISION}")
    print(f"Sequence lengths: {SEQ_LENGTHS}")
    print(f"Texts per length: {TEXTS_PER_LENGTH}")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Output dir: {FIXTURE_DIR}")
    print()

    corpus = load_corpus()
    print(f"Loaded {len(corpus)} corpus texts.")

    manifest = {
        "model_revision": REPO_REVISION,
        "repo_id": REPO_ID,
        "seq_lengths": SEQ_LENGTHS,
        "texts_per_length": TEXTS_PER_LENGTH,
        "files": [],
    }

    for seq in SEQ_LENGTHS:
        tag = f"{seq:04d}"
        print(f"\n=== seq_length={seq} ===")

        texts = select_texts(corpus, seq, TEXTS_PER_LENGTH)
        dense, sparse = compute_embeddings(texts, seq, CACHE_DIR)

        assert dense.shape == (TEXTS_PER_LENGTH, 1024), f"Unexpected dense shape: {dense.shape}"
        assert len(sparse) == TEXTS_PER_LENGTH

        # Write texts.
        texts_path = FIXTURE_DIR / f"texts_seq_{tag}.json"
        texts_bytes = json.dumps(texts, indent=2, ensure_ascii=False).encode("utf-8")
        texts_path.write_bytes(texts_bytes)

        # Write dense embeddings as npy.
        dense_path = FIXTURE_DIR / f"reference_dense_seq_{tag}.npy"
        np.save(str(dense_path), dense)

        # Write sparse embeddings as JSON.
        sparse_path = FIXTURE_DIR / f"reference_sparse_seq_{tag}.json"
        sparse_bytes = json.dumps(sparse, indent=2).encode("utf-8")
        sparse_path.write_bytes(sparse_bytes)

        manifest["files"].append({
            "seq_length": seq,
            "texts": texts_path.name,
            "dense": dense_path.name,
            "sparse": sparse_path.name,
            "sha256": {
                "texts": sha256_bytes(texts_bytes),
                "dense": sha256_file(dense_path),
                "sparse": sha256_bytes(sparse_bytes),
            },
        })

        print(f"  dense shape: {dense.shape}, dtype: {dense.dtype}")
        print(f"  sparse: {len(sparse)} texts, avg {np.mean([len(s['indices']) for s in sparse]):.1f} active tokens")
        print(f"  Written: {texts_path.name}, {dense_path.name}, {sparse_path.name}")

    # Write manifest.
    manifest_path = FIXTURE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {manifest_path}")

    # Write README.
    readme_path = FIXTURE_DIR / "README.md"
    readme_path.write_text(f"""\
# Equivalence Test Fixtures

Reference embeddings produced by `scripts/generate_equivalence_fixtures.py`
using the BAAI/bge-m3 FP32 model at revision `{REPO_REVISION}`.

## Files

| File | Description |
|------|-------------|
| `manifest.json` | Metadata, model revision, SHA-256 checksums |
| `texts_seq_NNNN.json` | Input texts (truncated to ~NNNN tokens) |
| `reference_dense_seq_NNNN.npy` | Dense embeddings `[N, 1024] f32`, L2-normalized |
| `reference_sparse_seq_NNNN.json` | Sparse embeddings `[N] {{indices, values}}` |

## Regenerating

Run after bumping `REPO_REVISION` in `src/embedder.rs`:

```bash
pip install transformers torch onnxruntime numpy safetensors huggingface_hub
python scripts/generate_equivalence_fixtures.py
```

The generator must produce fixtures for the same revision stored in
`REPO_REVISION` inside `src/embedder.rs`. The Rust drift-detection test
(`embedder::tests::repo_revision_consistent_across_all_copies`) will fail
if they diverge.

## Tolerances (from `tests/equivalence.rs`)

| Model variant | Mean cosine sim | p5 cosine sim |
|---------------|----------------|---------------|
| FP32          | ≥ 0.99         | ≥ 0.97        |
| FP16          | ≥ 0.98         | ≥ 0.96        |
| INT8          | ≥ 0.95         | ≥ 0.93        |
""")

    print("Done.")


if __name__ == "__main__":
    main()

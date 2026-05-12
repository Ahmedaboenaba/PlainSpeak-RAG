"""
PlainSpeak RAG -- Phase 1: Data Collection & Knowledge Base Indexing
===================================================================
Stage 1: Load & Extract datasets
Stage 2: Clean & Deduplicate
Stage 3: Compute FK grades
Stage 4: Embed & Build FAISS index
Stage 5: Save deliverables
Stage 6: Verify

Deliverables:
  - knowledge_base.jsonl   (5000 rows, one JSON per line)
  - kb_index.faiss          (FAISS IndexFlatL2 of complex embeddings)
  - kb_metadata.pkl         (list of dicts: domain/source/fk_grade/simple)
  - data_stats.txt          (row counts per source, avg FK grade per domain)
  - build_index.py          (this script)
"""

import json
import logging
import pickle
import random
import time
import os
import sys
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QUOTAS = {
    "wiki_auto":    2000,
    "simple_wiki":  1500,
    "medeasi":      1000,
    "multilexnorm":  500,
}
TARGET_TOTAL = 5000
MAX_WORD_TOKENS = 512          # max words per sentence (proxy for 512 tokens)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBED_BATCH_SIZE = 256

OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "knowledge_base_raw.jsonl"
FINAL_JSONL  = OUTPUT_DIR / "knowledge_base.jsonl"
FAISS_INDEX  = OUTPUT_DIR / "kb_index.faiss"
METADATA_PKL = OUTPUT_DIR / "kb_metadata.pkl"
STATS_FILE   = OUTPUT_DIR / "data_stats.txt"

MEDICAL_KEYWORDS = {
    "patient", "diagnosis", "treatment", "clinical", "symptom", "disease",
    "therapy", "medical", "surgical", "hospital", "physician", "medication",
    "prescription", "pathology", "prognosis", "chronic", "acute",
}
LEGAL_KEYWORDS = {
    "plaintiff", "defendant", "court", "statute", "jurisdiction", "liability",
    "counsel", "verdict", "amendment", "arbitration", "litigation", "judicial",
    "prosecution", "attorney", "contractual", "tort", "felony",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "build_index.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =====================================================================
# STAGE 1 — Load & Extract
# =====================================================================

def detect_domain(text: str, default: str = "general") -> str:
    """Assign domain label based on keyword presence."""
    words = set(text.lower().split())
    med_hits = words & MEDICAL_KEYWORDS
    legal_hits = words & LEGAL_KEYWORDS
    if len(med_hits) >= 2:
        return "medical"
    if len(legal_hits) >= 2:
        return "legal"
    return default


def load_wiki_auto(quota: int) -> list[dict]:
    """
    Load wiki_auto dataset: aligned (complex, simple) Wikipedia sentences.
    Source: chaojiang06/wiki_auto, config=auto_full_no_split
    Columns: normal_sentence (complex), simple_sentence (simple)
    """
    log.info(f"Loading wiki_auto (quota={quota})...")
    from datasets import load_dataset

    ds = load_dataset(
        "GEM/wiki_auto_asset_turk",
        "wiki_auto_asset_turk",
        split="train",
        trust_remote_code=True,
    )
    log.info(f"  wiki_auto loaded: {len(ds)} rows available")

    pairs = []
    # Shuffle indices for random sampling
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in tqdm(indices, desc="wiki_auto: extracting pairs"):
        if len(pairs) >= quota:
            break
        row = ds[idx]
        # wiki_auto_asset_turk has 'source' (complex) and 'target' (simple)
        # Some versions have 'source'/'target', others 'normal_sentence'/'simple_sentence'
        complex_text = row.get("source", row.get("normal_sentence", ""))
        simple_text = row.get("target", row.get("simple_sentence", ""))

        if not complex_text or not simple_text:
            continue
        complex_text = str(complex_text).strip()
        simple_text = str(simple_text).strip()
        if not complex_text or not simple_text:
            continue
        # If target is a list, take first
        if isinstance(simple_text, list):
            simple_text = simple_text[0] if simple_text else ""
        if isinstance(complex_text, list):
            complex_text = complex_text[0] if complex_text else ""

        pairs.append({
            "complex": complex_text,
            "simple":  simple_text,
            "domain":  "general",
            "source":  "wiki_auto",
        })

    log.info(f"  wiki_auto: extracted {len(pairs)} pairs")
    return pairs


def load_simple_wiki(quota: int) -> list[dict]:
    """
    Load sentence-transformers/simple-wiki dataset.
    Config: pair
    Columns: text (complex), simplified (simple)
    """
    log.info(f"Loading simple_wiki (quota={quota})...")
    from datasets import load_dataset

    ds = load_dataset(
        "sentence-transformers/simple-wiki",
        "pair",
        split="train",
        trust_remote_code=True,
    )
    log.info(f"  simple_wiki loaded: {len(ds)} rows available")

    pairs = []
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in tqdm(indices, desc="simple_wiki: extracting pairs"):
        if len(pairs) >= quota:
            break
        row = ds[idx]
        complex_text = str(row.get("text", "")).strip()
        simple_text  = str(row.get("simplified", "")).strip()
        if not complex_text or not simple_text:
            continue
        # Skip if complex == simple (no simplification)
        if complex_text == simple_text:
            continue

        pairs.append({
            "complex": complex_text,
            "simple":  simple_text,
            "domain":  "general",
            "source":  "simple_wiki",
        })

    log.info(f"  simple_wiki: extracted {len(pairs)} pairs")
    return pairs


def load_medeasi(quota: int) -> list[dict]:
    """
    Load Med-EASi dataset: medical expert-to-simple pairs.
    Source: cbasu/Med-EASi
    Columns: Expert (complex), Simple (simple)
    """
    log.info(f"Loading medeasi (quota={quota})...")
    from datasets import load_dataset

    ds = load_dataset("cbasu/Med-EASi")
    # Combine all splits
    all_rows = []
    for split_name in ds:
        log.info(f"  medeasi split '{split_name}': {len(ds[split_name])} rows")
        all_rows.extend(ds[split_name])
    log.info(f"  medeasi total: {len(all_rows)} rows available")

    pairs = []
    random.shuffle(all_rows)

    for row in tqdm(all_rows, desc="medeasi: extracting pairs"):
        if len(pairs) >= quota:
            break
        complex_text = str(row.get("Expert", "")).strip()
        simple_text  = str(row.get("Simple", "")).strip()
        if not complex_text or not simple_text:
            continue

        pairs.append({
            "complex": complex_text,
            "simple":  simple_text,
            "domain":  "medical",
            "source":  "medeasi",
        })

    log.info(f"  medeasi: extracted {len(pairs)} pairs")
    return pairs


def load_multilexnorm(quota: int) -> list[dict]:
    """
    Load MultiLexNorm dataset: lexical normalization (token-level).
    Source: MultiLexNorm, config=en, trust_remote_code=True
    Reconstruct sentences from token sequences.
    """
    log.info(f"Loading multilexnorm (quota={quota})...")
    from datasets import load_dataset

    ds = load_dataset("lexnorm", "en", trust_remote_code=True)
    log.info(f"  multilexnorm splits: {list(ds.keys())}")

    all_rows = []
    for split_name in ds:
        log.info(f"  multilexnorm split '{split_name}': {len(ds[split_name])} rows")
        all_rows.extend(ds[split_name])
    log.info(f"  multilexnorm total: {len(all_rows)} rows available")

    pairs = []
    random.shuffle(all_rows)

    for row in tqdm(all_rows, desc="multilexnorm: extracting pairs"):
        if len(pairs) >= quota:
            break

        # Token-level: reconstruct sentences
        # Expected fields: 'input' (list of tokens) and 'output'/'normalization' (list of normalized tokens)
        input_tokens = row.get("input", row.get("tokens", []))
        output_tokens = row.get("output", row.get("normalization", row.get("norm", [])))

        if isinstance(input_tokens, list):
            complex_text = " ".join(str(t) for t in input_tokens).strip()
        else:
            complex_text = str(input_tokens).strip()

        if isinstance(output_tokens, list):
            simple_text = " ".join(str(t) for t in output_tokens).strip()
        else:
            simple_text = str(output_tokens).strip()

        if not complex_text or not simple_text:
            continue
        # Skip if identical (no normalization happened)
        if complex_text == simple_text:
            continue

        domain = detect_domain(complex_text, default="general")

        pairs.append({
            "complex": complex_text,
            "simple":  simple_text,
            "domain":  domain,
            "source":  "multilexnorm",
        })

    log.info(f"  multilexnorm: extracted {len(pairs)} pairs")
    return pairs


# =====================================================================
# STAGE 2 — Clean & Deduplicate
# =====================================================================

def truncate_text(text: str, max_words: int = MAX_WORD_TOKENS) -> tuple[str, bool]:
    """Truncate text to max_words. Returns (text, was_truncated)."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]), True
    return text, False


def clean_and_deduplicate(all_pairs: list[dict]) -> list[dict]:
    """
    Stage 2: Clean empty rows, truncate long sentences, remove exact duplicates.
    """
    log.info(f"Stage 2: Cleaning {len(all_pairs)} total pairs...")

    # Step 1: Remove empty / whitespace-only rows
    cleaned = []
    empty_count = 0
    for p in all_pairs:
        if p["complex"].strip() and p["simple"].strip():
            cleaned.append(p)
        else:
            empty_count += 1
    log.info(f"  Removed {empty_count} empty rows -> {len(cleaned)} remaining")

    # Step 2: Truncate sentences > MAX_WORD_TOKENS
    trunc_complex = 0
    trunc_simple = 0
    for p in tqdm(cleaned, desc="Truncating long sentences"):
        p["complex"], was_trunc = truncate_text(p["complex"])
        if was_trunc:
            trunc_complex += 1
        p["simple"], was_trunc = truncate_text(p["simple"])
        if was_trunc:
            trunc_simple += 1
    log.info(f"  Truncated: {trunc_complex} complex, {trunc_simple} simple sentences")

    # Step 3: Deduplicate on exact 'complex' string match
    seen = set()
    deduped = []
    dup_count = 0
    for p in tqdm(cleaned, desc="Deduplicating"):
        if p["complex"] not in seen:
            seen.add(p["complex"])
            deduped.append(p)
        else:
            dup_count += 1
    log.info(f"  Removed {dup_count} duplicate 'complex' entries -> {len(deduped)} unique pairs")

    return deduped


# =====================================================================
# MAIN
# =====================================================================

def run_stages_1_2() -> list[dict]:
    """Stage 1: Load & Extract  +  Stage 2: Clean & Deduplicate."""
    random.seed(42)

    # ------------------------------------------------------------------
    # STAGE 1: Load & Extract
    # ------------------------------------------------------------------
    loaders = [
        ("wiki_auto",    load_wiki_auto,    QUOTAS["wiki_auto"]),
        ("simple_wiki",  load_simple_wiki,  QUOTAS["simple_wiki"]),
        ("medeasi",      load_medeasi,      QUOTAS["medeasi"]),
        ("multilexnorm", load_multilexnorm, QUOTAS["multilexnorm"]),
    ]

    all_pairs: list[dict] = []
    failed_sources: list[str] = []
    source_counts: dict[str, int] = {}

    for name, loader_fn, quota in loaders:
        try:
            pairs = loader_fn(quota)
            all_pairs.extend(pairs)
            source_counts[name] = len(pairs)
        except Exception as e:
            log.error(f"FAILED to load {name}: {e}", exc_info=True)
            failed_sources.append(name)
            source_counts[name] = 0

    log.info("-" * 40)
    log.info("Stage 1 Summary:")
    for src, cnt in source_counts.items():
        status = "[OK]" if src not in failed_sources else "[FAIL]"
        log.info(f"  {src:15s}: {cnt:5d} pairs  {status}")
    log.info(f"  {'TOTAL':15s}: {len(all_pairs):5d} pairs")

    # Redistribute quota if any source failed
    if failed_sources:
        deficit = sum(QUOTAS[s] for s in failed_sources)
        active_sources = [s for s in QUOTAS if s not in failed_sources and source_counts[s] > 0]
        if active_sources and deficit > 0:
            extra_per_source = deficit // len(active_sources)
            remainder = deficit % len(active_sources)
            log.info(f"  Redistributing {deficit} from failed sources across {active_sources}")
            log.info(f"  +{extra_per_source} each, +{remainder} to first source")

            for i, name in enumerate(active_sources):
                _, loader_fn, _ = next(t for t in loaders if t[0] == name)
                extra = extra_per_source + (remainder if i == 0 else 0)
                new_quota = source_counts[name] + extra
                try:
                    extra_pairs = loader_fn(new_quota)
                    existing_complex = {p["complex"] for p in all_pairs}
                    new_pairs = [p for p in extra_pairs if p["complex"] not in existing_complex]
                    all_pairs.extend(new_pairs[:extra])
                    source_counts[name] += len(new_pairs[:extra])
                    log.info(f"  {name}: added {len(new_pairs[:extra])} extra pairs")
                except Exception as e:
                    log.error(f"  Failed to get extra from {name}: {e}")

    # ------------------------------------------------------------------
    # STAGE 2: Clean & Deduplicate
    # ------------------------------------------------------------------
    clean_pairs = clean_and_deduplicate(all_pairs)

    if len(clean_pairs) > TARGET_TOTAL:
        log.info(f"  Trimming {len(clean_pairs)} -> {TARGET_TOTAL}")
        random.shuffle(clean_pairs)
        clean_pairs = clean_pairs[:TARGET_TOTAL]
    elif len(clean_pairs) < TARGET_TOTAL:
        log.warning(
            f"  Only {len(clean_pairs)} pairs available (target={TARGET_TOTAL}). "
            f"Proceeding with fewer rows."
        )

    # Save intermediate JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in tqdm(clean_pairs, desc="Writing knowledge_base_raw.jsonl"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(clean_pairs)} rows to {OUTPUT_FILE}")

    return clean_pairs


def load_raw_jsonl() -> list[dict]:
    """Load pairs from the previously saved raw JSONL."""
    log.info(f"Loading existing {OUTPUT_FILE.name} ...")
    pairs = []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    log.info(f"  Loaded {len(pairs)} pairs from cache")
    return pairs


# =====================================================================
# STAGE 3 -- Compute FK Grades
# =====================================================================

def compute_fk_grades(pairs: list[dict]) -> list[dict]:
    """Add fk_grade (Flesch-Kincaid grade of 'simple' text) to each pair."""
    import textstat

    log.info("Stage 3: Computing Flesch-Kincaid grades...")
    for p in tqdm(pairs, desc="Computing FK grades"):
        try:
            fk = textstat.flesch_kincaid_grade(p["simple"])
            # Clamp negative values (very short texts) to 0.0
            p["fk_grade"] = max(0.0, round(fk, 2))
        except Exception:
            p["fk_grade"] = 0.0

    # Sanity stats
    grades = [p["fk_grade"] for p in pairs]
    log.info(f"  FK grades: min={min(grades):.1f}, max={max(grades):.1f}, "
             f"mean={sum(grades)/len(grades):.2f}")
    return pairs


# =====================================================================
# STAGE 4 -- Embed & Build FAISS Index
# =====================================================================

def embed_and_index(pairs: list[dict]):
    """
    Encode all 'complex' texts with all-MiniLM-L6-v2 and build FAISS IndexFlatL2.
    Returns (index, embeddings_array).
    """
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

    log.info(f"Stage 4: Embedding {len(pairs)} texts with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    complex_texts = [p["complex"] for p in pairs]
    embeddings = model.encode(
        complex_texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)
    log.info(f"  Embeddings shape: {embeddings.shape}")

    # Build FAISS index
    log.info(f"  Building FAISS IndexFlatL2 (dim={EMBEDDING_DIM})...")
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    log.info(f"  Index built: ntotal={index.ntotal}")

    return index, embeddings


# =====================================================================
# STAGE 5 -- Save Deliverables
# =====================================================================

def save_deliverables(pairs: list[dict], index):
    """Save knowledge_base.jsonl, kb_index.faiss, kb_metadata.pkl, data_stats.txt."""
    import faiss

    log.info("Stage 5: Saving deliverables...")

    # 1. knowledge_base.jsonl
    with open(FINAL_JSONL, "w", encoding="utf-8") as f:
        for row in tqdm(pairs, desc="Writing knowledge_base.jsonl"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"  Saved {FINAL_JSONL.name} ({FINAL_JSONL.stat().st_size / 1024:.1f} KB)")

    # 2. kb_index.faiss
    faiss.write_index(index, str(FAISS_INDEX))
    log.info(f"  Saved {FAISS_INDEX.name} ({FAISS_INDEX.stat().st_size / 1024 / 1024:.2f} MB)")

    # 3. kb_metadata.pkl  (row-aligned with FAISS index)
    metadata = [
        {
            "domain":   p["domain"],
            "source":   p["source"],
            "fk_grade": p["fk_grade"],
            "simple":   p["simple"],
        }
        for p in pairs
    ]
    with open(METADATA_PKL, "wb") as f:
        pickle.dump(metadata, f)
    log.info(f"  Saved {METADATA_PKL.name} ({METADATA_PKL.stat().st_size / 1024:.1f} KB)")

    # 4. data_stats.txt
    source_counts: dict[str, int] = {}
    domain_fk: dict[str, list[float]] = {}
    for p in pairs:
        source_counts[p["source"]] = source_counts.get(p["source"], 0) + 1
        domain_fk.setdefault(p["domain"], []).append(p["fk_grade"])

    lines = [
        "=== PlainSpeak RAG Knowledge Base Stats ===",
        f"Total rows: {len(pairs)}",
        "",
        "Rows per source:",
    ]
    for src, cnt in sorted(source_counts.items()):
        lines.append(f"  {src:15s}: {cnt}")
    lines.append("")
    lines.append("Avg FK grade per domain:")
    for dom, grades in sorted(domain_fk.items()):
        avg = sum(grades) / len(grades)
        lines.append(f"  {dom:15s}: {avg:.2f}")
    lines.append("")
    lines.append(f"FAISS index vectors: {index.ntotal}")
    lines.append(f"Embedding dimension: {EMBEDDING_DIM}")
    lines.append(f"Embedding model:     {EMBEDDING_MODEL}")

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"  Saved {STATS_FILE.name}")


# =====================================================================
# STAGE 6 -- Verify
# =====================================================================

def verify(pairs: list[dict]):
    """Reload index, run 10 random queries, measure avg latency."""
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

    log.info("Stage 6: Verification...")

    # Reload index from disk
    index = faiss.read_index(str(FAISS_INDEX))
    log.info(f"  Reloaded index: ntotal={index.ntotal}")

    # Reload metadata from disk
    with open(METADATA_PKL, "rb") as f:
        metadata = pickle.load(f)
    log.info(f"  Reloaded metadata: {len(metadata)} rows")

    # Check counts
    assert index.ntotal == len(pairs), (
        f"Index ntotal ({index.ntotal}) != pairs ({len(pairs)})"
    )
    assert len(metadata) == len(pairs), (
        f"Metadata len ({len(metadata)}) != pairs ({len(pairs)})"
    )

    # Schema validation on JSONL
    required_keys = {"complex", "simple", "domain", "source", "fk_grade"}
    with open(FINAL_JSONL, "r", encoding="utf-8") as f:
        jsonl_count = 0
        for line in f:
            row = json.loads(line)
            missing = required_keys - set(row.keys())
            assert not missing, f"Row missing keys: {missing}"
            assert 0 <= row["fk_grade"] <= 100, f"FK grade out of range: {row['fk_grade']}"
            jsonl_count += 1
    log.info(f"  JSONL schema check passed: {jsonl_count} rows, all 5 keys present")

    # Duplicate check
    complexes = set()
    dup = 0
    with open(FINAL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)["complex"]
            if c in complexes:
                dup += 1
            complexes.add(c)
    log.info(f"  Duplicate check: {dup} duplicates found")

    # Query latency test (10 random queries)
    model = SentenceTransformer(EMBEDDING_MODEL)
    random.seed(99)
    sample_indices = random.sample(range(len(pairs)), min(10, len(pairs)))
    sample_texts = [pairs[i]["complex"] for i in sample_indices]

    latencies = []
    for text in sample_texts:
        q_emb = model.encode([text], convert_to_numpy=True).astype(np.float32)
        t0 = time.perf_counter()
        D, I = index.search(q_emb, 5)  # top-5
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    avg_lat = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    log.info(f"  Query latency (10 queries, top-5):")
    log.info(f"    avg = {avg_lat:.2f} ms")
    log.info(f"    max = {max_lat:.2f} ms")
    if avg_lat < 200:
        log.info(f"    [PASS] avg latency < 200 ms")
    else:
        log.warning(f"    [WARN] avg latency >= 200 ms")

    # Print final summary
    log.info("=" * 60)
    log.info("FINAL PIPELINE SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total rows:        {len(pairs)}")
    log.info(f"  FAISS index size:  {FAISS_INDEX.stat().st_size / 1024 / 1024:.2f} MB")
    log.info(f"  Metadata size:     {METADATA_PKL.stat().st_size / 1024:.1f} KB")
    log.info(f"  JSONL size:        {FINAL_JSONL.stat().st_size / 1024:.1f} KB")
    log.info(f"  Avg query latency: {avg_lat:.2f} ms")
    log.info(f"  Duplicates:        {dup}")
    log.info("  Deliverables:")
    for f in [FINAL_JSONL, FAISS_INDEX, METADATA_PKL, STATS_FILE]:
        log.info(f"    {f.name:25s} {'EXISTS' if f.exists() else 'MISSING'}")
    log.info("=" * 60)


# =====================================================================
# MAIN
# =====================================================================

def main():
    log.info("=" * 60)
    log.info("PlainSpeak RAG -- Phase 1: Full Pipeline")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Stages 1-2: Load & Clean (skip if raw file exists)
    # ------------------------------------------------------------------
    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        log.info(f"  {OUTPUT_FILE.name} already exists, skipping Stages 1-2.")
        pairs = load_raw_jsonl()
    else:
        pairs = run_stages_1_2()

    log.info(f"  Working with {len(pairs)} pairs")

    # ------------------------------------------------------------------
    # Stage 3: FK Grades
    # ------------------------------------------------------------------
    pairs = compute_fk_grades(pairs)

    # ------------------------------------------------------------------
    # Stage 4: Embed & Index
    # ------------------------------------------------------------------
    index, embeddings = embed_and_index(pairs)

    # ------------------------------------------------------------------
    # Stage 5: Save Deliverables
    # ------------------------------------------------------------------
    save_deliverables(pairs, index)

    # ------------------------------------------------------------------
    # Stage 6: Verify
    # ------------------------------------------------------------------
    verify(pairs)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()

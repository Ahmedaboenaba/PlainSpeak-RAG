import json
import pickle
import time
import logging
import faiss
import numpy as np
import spacy
from pathlib import Path
from sentence_transformers import SentenceTransformer
from textstat import flesch_kincaid_grade

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

class PlainSpeakRetrieval:
    def __init__(self, 
                 index_path="kb_index.faiss", 
                 metadata_path="kb_metadata.pkl", 
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
        
        self.root = Path(__file__).parent
        self.index_path = self.root / index_path
        self.metadata_path = self.root / metadata_path
        
        log.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        log.info(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(str(self.index_path))
        
        log.info(f"Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        log.info("Loading spaCy pipeline...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            log.warning("en_core_web_sm not found, downloading...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        # Define Keywords for Domain Detection
        self.medical_keywords = {
            "patient", "diagnosis", "treatment", "clinical", "symptom", "disease",
            "therapy", "medical", "surgical", "hospital", "physician", "medication",
            "prescription", "pathology", "prognosis", "chronic", "acute", "myocardial",
            "infarction", "cardiology", "neurology", "oncology", "diabetes", "hypertension",
            "malignant", "tumor", "cancer", "surgery", "operation", "biopsy"
        }
        self.legal_keywords = {
            "plaintiff", "defendant", "court", "statute", "jurisdiction", "liability",
            "counsel", "verdict", "amendment", "arbitration", "litigation", "judicial",
            "prosecution", "attorney", "contractual", "tort", "felony", "misdemeanor",
            "testimony", "affidavit", "subpoena", "judge", "lawsuit", "legal"
        }

    def detect_domain(self, text: str) -> str:
        """Detect if the text is medical, legal, or general."""
        doc = self.nlp(text.lower())
        words = {token.text for token in doc}
        
        # Check entities
        entities = {ent.label_ for ent in doc.ents}
        
        med_hits = words & self.medical_keywords
        legal_hits = words & self.legal_keywords
        
        # Priority logic
        if len(med_hits) >= 1 or any(label in ["DISEASE", "CHEMICAL"] for label in entities):
            return "medical"
        if len(legal_hits) >= 1 or "LAW" in entities:
            return "legal"
        
        return "general"

    def get_jargon_score(self, text: str):
        """Identify potential jargon words based on syllable count and rarity (simple heuristic)."""
        doc = self.nlp(text)
        jargon_tokens = []
        for token in doc:
            # Heuristic: long words in non-stopword categories
            if not token.is_stop and not token.is_punct and len(token.text) > 8:
                jargon_tokens.append(token.text)
        return jargon_tokens

    def retrieve(self, query: str, target_grade: float, top_k: int = 10):
        """
        1. Detect domain
        2. Retrieve top_k from FAISS
        3. Filter by domain
        4. Re-rank by |fk_grade - target_grade|
        """
        t0 = time.perf_counter()
        
        # 1. Detect Domain
        domain = self.detect_domain(query)
        log.info(f"Detected domain: {domain}")
        
        # 2. Embed & Search
        query_vec = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        
        # Search for more than top_k to allow for filtering
        distances, indices = self.index.search(query_vec, k=top_k * 5)
        
        # 3. Filter and Prepare Candidates
        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            meta = self.metadata[idx]
            
            # Domain Filter: match domain or fallback to general if no matches found
            if meta["domain"] == domain or domain == "general":
                candidates.append({
                    "complex": meta["complex"],
                    "simple": meta["simple"],
                    "fk_grade": meta["fk_grade"],
                    "domain": meta["domain"],
                    "source": meta["source"],
                    "distance": dist
                })
        
        # If no domain-specific matches, just take the top-k raw
        if not candidates:
            log.warning(f"No domain-specific matches for '{domain}', using raw top results.")
            for dist, idx in zip(distances[0][:top_k], indices[0][:top_k]):
                meta = self.metadata[idx]
                candidates.append({
                    "complex": meta["complex"],
                    "simple": meta["simple"],
                    "fk_grade": meta["fk_grade"],
                    "domain": meta["domain"],
                    "source": meta["source"],
                    "distance": dist
                })

        # 4. Re-rank by FK Grade proximity
        # Closest to target_grade wins
        candidates.sort(key=lambda x: abs(x["fk_grade"] - target_grade))
        
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        
        return {
            "best_match": candidates[0] if candidates else None,
            "top_candidates": candidates[:3],
            "domain": domain,
            "latency_ms": latency_ms
        }

if __name__ == "__main__":
    # Quick sanity check
    engine = PlainSpeakRetrieval()
    
    test_queries = [
        ("The patient exhibits symptoms of myocardial infarction.", 5.0),
        ("The defendant is liable for breach of contract.", 8.0),
        ("A large asteroid is moving towards the Earth's orbit.", 5.0)
    ]
    
    for query, grade in test_queries:
        print(f"\nQuery: {query} (Target: Grade {grade})")
        result = engine.retrieve(query, grade)
        print(f"Domain: {result['domain']}")
        print(f"Latency: {result['latency_ms']:.2f}ms")
        if result['best_match']:
            print(f"Best Simple: {result['best_match']['simple']} (FK: {result['best_match']['fk_grade']})")

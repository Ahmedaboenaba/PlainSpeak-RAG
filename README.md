# PlainSpeak RAG: Domain-Grounded Text Simplification

PlainSpeak RAG is a Retrieval-Augmented Generation (RAG) system designed to simplify jargon-heavy text (medical, legal, general) while preserving meaning. It uses domain-filtered retrieval and Flesch-Kincaid re-ranking to ground LLM generation in verified plain-language examples.

## 🚀 Project Overview
The core problem PlainSpeak RAG solves is that generic LLM simplification often loses domain-specific nuances or fails to adhere to a specific reading level. By retrieving real-world (Complex, Simple) pairs from relevant domains, we provide the LLM with high-quality few-shot examples that match the desired output style and complexity.

### Key Features
- **Domain-Aware Retrieval**: Automatically detects if input is `medical`, `legal`, or `general` to fetch relevant examples.
* **Reading Level Control**: Re-ranks search results based on Flesch-Kincaid (FK) grade targets (Grade 5 / Grade 8 / Expert).
* **High-Performance Vector Search**: Uses FAISS for sub-millisecond retrieval across ~5,000 text pairs.

---

## 🛠 Tech Stack
- **NLP**: spaCy 3.x (Domain Detection, NER)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Vector DB**: FAISS (IndexFlatL2)
- **Readability**: `textstat` (Flesch-Kincaid Grade)
- **Python Version**: 3.10 (Recommended for stability)

---

## 📁 Repository Structure
- `build_index.py`: Data engineering pipeline (Extraction -> Cleaning -> Embedding -> Indexing).
- `retrieval_engine.py`: Core logic for domain detection and smart search.
- `test_retrieval.py`: Verification suite for latency and accuracy.
- `knowledge_base.jsonl`: The processed dataset of 4,987 text pairs.
- `kb_index.faiss`: The vector index for similarity search.
- `kb_metadata.pkl`: Aligned metadata (source, domain, FK grade) for re-ranking.
- `data_stats.txt`: Summary statistics of the knowledge base.

---

## 📊 Knowledge Base Stats
Current distribution of the **4,987** text pairs:

| Source | Count | Domain |
| :--- | :--- | :--- |
| **Wiki-Auto** | 2,168 | General |
| **Simple-Wiki** | 1,665 | General |
| **Med-EASi** | 1,154 | Medical |

**Average Complexity (FK Grade):**
- General Domain: **9.23**
- Medical Domain: **11.74**

---

## 🚦 Getting Started

### 1. Prerequisites
Ensure you are using Python 3.10 and install the dependencies:
```bash
pip install datasets sentence-transformers faiss-cpu textstat tqdm spacy
python -m spacy download en_core_web_sm
```

### 2. Build/Update the Index
To rebuild the knowledge base and FAISS index:
```bash
python build_index.py
```

### 3. Run Retrieval Tests
Verify the engine's performance (Latency & Domain Detection):
```bash
python test_retrieval.py
```

---

## 🧠 Core Components

### Data Engineering (`build_index.py`)
Processes data in 6 stages:
1. **Extraction**: Loads pairs from Wikipedia and Med-EASi.
2. **Cleaning**: Deduplicates sentences and truncates long inputs (>512 words).
3. **Readability**: Assigns an FK grade to every simplified pair.
4. **Embedding**: Vectorizes complex sentences with MiniLM.
5. **Indexing**: Builds a FAISS index for L2 distance search.
6. **Verification**: Validates index integrity and latency.

### Retrieval Engine (`retrieval_engine.py`)
1. **Domain Detection**: Uses a hybrid system of spaCy NER and keyword density (e.g., "myocardial" -> Medical).
2. **Top-K Search**: Fetches the top-K semantically similar examples.
3. **FK Re-ranking**: Re-sorts candidates to find the one closest to the user's requested grade level.

---

## 📈 Performance
- **Query Latency**: ~30ms (CPU)
- **Domain Accuracy**: 80%+
- **Index Search Latency**: 0.65ms
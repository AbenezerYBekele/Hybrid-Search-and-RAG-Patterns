# 🧠 Hybrid search & RAG Patterns

> A modular, end-to-end Python reference for building production-ready Retrieval-Augmented Generation (RAG) pipelines — covering document ingestion, hybrid search, cost optimization, and dynamic model fallback.

---

## 📌 Overview

This repository walks through the full LLM application stack in self-contained, runnable scripts. Each module isolates a specific concept — from raw embeddings and vector stores to Hybrid Search with Reciprocal Rank Fusion (RRF) — so you can study, extend, or drop any component into your own project.

**Key technologies:** LangChain · ChromaDB · BM25 · HuggingFace Embeddings · Groq · Google Gemini

---

## 🗂️ Repository Structure

```
├── main.py                  # Environment verification & API testing (Groq & Gemini)
├── raw_emb.py               # Chat completion validation using system/user messages
│
├── document_loaders.py      # Ingestion handlers for raw Text and PDF documents
├── text_slitters.py         # Analysis of text splitting and chunk overlap parameters
│
├── embeddings_deep.py       # Local embeddings evaluation and manual cosine similarity ranking
├── app.py                   # Simple transient document indexing with ChromaDB
├── vector_stores.py         # Chroma persistence, distance scoring, and metadata filtering
│
├── hybrid_retriever.py      # Production-style Hybrid Search (BM25 + Chroma) using RRF
├── hybrid_test.py           # Evaluation script comparing Hybrid vs. standalone retrievers
│
├── rag_pipeline.py          # Complete RAG pipeline with document splitting and model fallbacks
└── cost_optimization.py     # Cost management: dynamic routing, caching, and token budgeting
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites

Python 3.9 or higher is required. A virtual environment is strongly recommended.

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install Dependencies

Create a `requirements.txt` with the following packages:

```
chromadb
langchain-chroma
langchain-core
langchain-huggingface
langchain-groq
langchain-google-genai
langchain-community
transformers
pypdf
numpy
python-dotenv
torch
rank-bm25
```

Then install:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY="your_groq_api_key_here"
GOOGLE_API_KEY="your_google_api_key_here"
```

---

## 🧩 Component Breakdown

### 1. Validation & API Testing

| File | Description |
|------|-------------|
| `main.py` | Verifies installed packages (`langchain-core`, `langgraph`) and tests connectivity to Groq (`llama-3.1-8b-instant`) and Google Gemini (`gemini-2.0-flash`) |
| `raw_emb.py` | Validates chat mechanics using LangChain message schemas (`SystemMessage`, `HumanMessage`) with Groq |

### 2. Document Processing & Ingestion

| File | Description |
|------|-------------|
| `document_loaders.py` | Ingests flat text files and multi-page PDFs via `TextLoader` and `PyPDFLoader` |
| `text_slitters.py` | Benchmarks chunking strategies using `RecursiveCharacterTextSplitter`, comparing non-overlapping vs. overlapping splits for semantic preservation |

### 3. Embeddings & Vector Databases

| File | Description |
|------|-------------|
| `embeddings_deep.py` | Evaluates the local `all-MiniLM-L6-v2` model, manually computes cosine similarity with NumPy, and ranks document relevancy |
| `app.py` | Rapid in-memory ChromaDB collection setup and querying (transient client) |
| `vector_stores.py` | Persistent Chroma index on disk, raw distance score analysis, and metadata key-value filtering |

### 4. Hybrid Search with Reciprocal Rank Fusion

| File | Description |
|------|-------------|
| `hybrid_retriever.py` | `HybridRetriever` class running BM25 (lexical) and Chroma (dense vector) in parallel, merged via RRF. Supports dynamic index updates at runtime |
| `hybrid_test.py` | Side-by-side evaluation of hybrid vs. standalone retrievers across exact SKUs, error codes, and natural-language queries |

### 5. Fallback RAG Pipeline

| File | Description |
|------|-------------|
| `rag_pipeline.py` | Full RAG loop: document splitting → Chroma retrieval → context injection → LLM generation, with sequential model API failover to maintain availability, assembled in a LangChain Expression Language (LCEL) chain |

### 6. Production Cost Optimization

| File | Description |
|------|-------------|
| `cost_optimization.py` | Three cost-control patterns: **Model Routing** (complexity-based dispatch to small vs. large models), **Normalized Caching** (deduplicating equivalent prompts before generation), and **Token Budgeting** (pre-flight rejection of over-limit prompts) |

---

## 🔍 Deep Dive: Hybrid Retrieval & RRF

Dense vector search alone fails on alphanumeric identifiers — product SKUs like `SKU-7742X` or error codes like `E_CONN_REFUSED` are poorly represented in embedding space. BM25 handles exact keyword matches perfectly but has no conceptual understanding.

**Hybrid Search** runs both methods in parallel and merges their ranked lists using **Reciprocal Rank Fusion (RRF)**:

$$\text{RRF}(d) = \sum_{r \in R} \frac{w_r}{k + \text{rank}_r(d)}$$

| Symbol | Meaning |
|--------|---------|
| `R` | Set of retrievers (BM25 + Vector Search) |
| `w_r` | Weight multiplier assigned to retriever `r` |
| `rank_r(d)` | Position of document `d` in retriever `r`'s results |
| `k` | Smoothing constant (typically `60`) preventing top-ranked docs from dominating |

This bypasses the problem of reconciling incompatible scoring systems (TF-IDF scores vs. cosine distances) and produces a single, stable relevance ranking.

---

## 🚀 Execution Guide

Run scripts sequentially from the project root with your virtual environment active:

```bash
# 1. Verify environment and test API connectivity
python main.py
python raw_emb.py

# 2. Explore document loading and splitting behavior
python document_loaders.py
python text_slitters.py

# 3. Analyze embeddings and vector database strategies
python embeddings_deep.py
python app.py
python vector_stores.py

# 4. Compare hybrid retrieval against standalone search
python hybrid_retriever.py
python hybrid_test.py

# 5. Run a full RAG loop with dynamic model fallback
python rag_pipeline.py

# 6. Test cost-optimization rules at runtime
python cost_optimization.py
```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

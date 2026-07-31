# 📝 Case Study: Optimizing RAG Pipelines
### Resolving Retrieval Bottlenecks, Silent Index Failures, and Selection Bias

This case study documents the engineering journey of optimizing a Retrieval-Augmented Generation (RAG) system built over the official FastAPI English documentation. It analyzes how chunking boundaries and secondary reranking affect generation metrics, and highlights three critical bugs encountered and resolved during development.

---

## 1. The Core Retrieval Bottleneck

In RAG systems, generation quality is strictly bounded by retrieval recall. Under a standard baseline pipeline using $top\_k=5$ candidate retrieval:
* **Fixed-Size Chunking** (character-window sliding) yielded low retrieval recall.
* **Structure-Aware** and **Semantic** chunking strategies split documents conceptually, but their recall and precision remained extremely low. 
* Many queries returned relevant information split across disparate chunks, resulting in incomplete context. This retrieval bottleneck caused downstream generation metrics (faithfulness and correctness) to degrade as the generator struggled to assemble answers from fragmented sources.

To resolve this, we implemented **Cross-Encoder Reranking** using `cross-encoder/ms-marco-MiniLM-L-6-v2`. By querying a deeper candidate pool ($top\_k=20$ from dense and sparse search combined via Reciprocal Rank Fusion) and reranking them down to the top 5, we decoupled candidate matching from context slot constraints.

---

## 2. Diagnostics: Three Critical Engineering Bugs Resolved

During the implementation and benchmarking phases, three silent failures occurred that would have corrupted the comparison results if left undetected.

### Bug 1: Silent Batch-Embedding Mismatch (Python `zip()` Truncation)
* **The Symptom**: Upon querying the ChromaDB collections, the total chunk count was far lower than expected for the size of the crawled raw documentation corpus.
* **The Cause**: The Gemini Embedding API was called in batches. Due to a model response format mismatch (where the API returned a structure where the list of embeddings was shorter than the list of source chunk texts), Python's built-in `zip(chunks, embeddings)` function silently stopped processing elements when the shorter list (embeddings) was exhausted.
* **The Diagnosis & Fix**: The issue was caught by manually querying the collection count and comparing it to the chunk list length. We resolved it by wrapping the batch loop in rigorous list-length validations before merging and replacing the standard `zip()` call with list-length assertions to ensure 100% of chunks are embedded and indexed.

### Bug 2: Silent BM25 Ingestion Failures
* **The Symptom**: During a evaluation run, the Confidence Guard blocked generation for almost all queries, claiming that top retrieval scores were below the `0.025` threshold.
* **The Cause**: The sparse (BM25) retriever was silently failing to locate the pickled index files (`bm25_*.pkl`). Rather than raising an error, it returned an empty results list. Because only the dense retriever was contributing to the Reciprocal Rank Fusion (RRF), the resulting RRF scores were mathematically halved (around `0.015`), pushing them below the Confidence Guard threshold.
* **The Diagnosis & Fix**: Diagnosed by plotting the distribution of RRF retrieval scores and noticing a step-drop. We wrote a dedicated integrity verification script (`verify_indexes.py`) that checks the byte-size of the pickled index files and asserts that the BM25 document count matches ChromaDB collection size. The BM25 indexing pipeline was updated to throw descriptive errors on missing files.

### Bug 3: Evaluation Selection Bias
* **The Symptom**: The pre-reranker run showed high **Citation Accuracy (74.33%)** and **Faithfulness (76.33%)**, but these dropped sharply to **27.33%** and **21.94%** respectively once the Reranker was activated.
* **The Cause**: This was a classic selection bias artifact. Without the Reranker, the retrieval scores were low, causing the Confidence Guard to block 42 of the 52 evaluation queries (marking them as ungrounded/insufficient context). The metrics were only evaluated over the remaining 10 "easy" queries. When the Reranker was activated, it successfully retrieved high-quality contexts for **30 queries (a 3x increase)**.
* **The Diagnosis & Fix**: Diagnosed by inspecting the evaluation report JSON files and counting the number of queries that returned non-`None` scores. Exposing the generator to 3x more queries introduced more complex, harder multi-hop and ambiguous queries, which naturally lowered the overall average citation accuracy while providing a much more honest and representative benchmark.

---

## 3. Reranker Impact & Ingestion Stats

Introducing the Cross-Encoder Reranker completely transformed retrieval performance across all three strategies:

* **Retrieval Recall Boost**: For the `structure_aware` strategy, Retrieval Recall rose from **23.08%** to **83.65%** (a **262% relative increase**).
* **Retrieval Precision Boost**: Retrieval Precision rose from **0.0%** (where zero relevant documents were returned under top_k limit constraints) to **28.04%** for `structure_aware` and **29.52%** for `fixed_size`.

---

## 4. Final Chunking Strategy Comparison

The final evaluation benchmark over the 52 golden queries yielded the following comparative results (Reranker active):

| Metric | Fixed-Size | Structure-Aware | Semantic |
| :--- | :---: | :---: | :---: |
| **Retrieval Recall** | 0.8141 | **0.8365** | 0.7596 |
| **Retrieval Precision** | **0.2952** | 0.2804 | 0.2641 |
| **Citation Accuracy** | 0.2733 | 0.3718 | **0.4340** |
| **Faithfulness** | 0.2194 | **0.3205** | 0.2807 |
| **Answer Correctness** | **0.4712** | 0.4269 | 0.3625 |

### Trade-offs & Engineering Recommendations

1. **Structure-Aware Chunking (Recommended Default)**:
   * By partitioning chunks along markdown headers (`#`, `##`, etc.) and prepending heading metadata to the text, this strategy preserves local hierarchies. 
   * It achieved the highest **Retrieval Recall (83.65%)** and **Faithfulness (32.05%)**, making it the best default strategy for structured technical manuals.
2. **Semantic Chunking**:
   * Uses sentence embeddings to split text when semantic similarity drops below a threshold. 
   * It achieved the highest **Citation Accuracy (43.40%)**, demonstrating that semantic boundaries align well with how language models formulate and cite single-focus claims, but it suffered from a slightly lower recall.
3. **Fixed-Size Chunking**:
   * Character-window slicing is fast to compute, but splits concepts in half.
   * Although it scored highly on simple lookups, its faithfulness remained lowest (**21.94%**), as context truncation frequently forces the generator to hallucinate missing connections.

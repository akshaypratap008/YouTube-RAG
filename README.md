# YouTube-RAG

Ask questions about any YouTube video and get answers grounded in its transcript, with timestamps pointing to where the answer appears.

The pipeline fetches the transcript, splits it into semantic chunks, embeds them with OpenAI, and stores them in a separate FAISS index for each video. At query time it retrieves the top-k chunks and has `gpt-4o-mini` answer using only that context. The workflow runs as a **LangGraph** state graph traced in **LangSmith**, and the retriever and generator are evaluated separately and end to end with **DeepEval** and LangSmith LLM-as-judge evaluators.

---

## How it works

```
YouTube URL ──► extract video_id ──► FAISS index exists for this video?
                                         │
                       no ◄──────────────┴──────────────► yes
                       │                                    │
   fetch transcript (youtube-transcript-api)                │
                       │                                    │
   merge short segments (min 10 words)                      │
                       │                                    │
   semantic chunking (all-MiniLM-L6-v2 similarity)          │
                       │                                    │
   embed chunks (text-embedding-3-large)                    │
                       │                                    │
   build + persist FAISS index  faiss_store/<video_id>/     │
                       │                                    │
                       └──────────────► similarity search (top_k=5) ◄┘
                                                  │
                                 gpt-4o-mini answer (context-only prompt)
                                                  │
                                    answer + (start, end) timestamps
```

### Pipeline stages

| Stage | Component | Details |
|---|---|---|
| Ingestion | `DataLoader.fetch_video_data` | Pulls the transcript with `youtube-transcript-api`, then merges short segments until each has at least 10 words, keeping `start`/`end` times. |
| Chunking | `DataLoader.create_semantic_chunks` | Embeds segments with `all-MiniLM-L6-v2` and starts a new chunk when cosine similarity to the previous segment falls below `0.40` **and** the current chunk is over `300` words. Consecutive chunks overlap by 1 segment. Every chunk keeps its `text`, `start` and `end`. |
| Embedding | `EmbeddingManager` | OpenAI `text-embedding-3-large` through `langchain-openai`. |
| Vector store | `FaissVectorStore` | `IndexFlatL2`, saved for each video as `faiss_store/<video_id>/faiss.index` plus `metadata.pkl` (the chunk dicts). |
| Retrieval | `RAGSearch.search` | Embeds the query and returns the top-k chunk texts. Results are cached on the instance so timestamps can be read later. |
| Generation | `RAGSearch.generate_response` | `gpt-4o-mini` at `temperature=0` with a strict context-only system prompt (`prompts/system_prompt.txt`). If the context doesn't contain the answer, it says so. |
| Timestamps | `RAGSearch.get_video_timestamps` | `(start, end)` for each retrieved chunk, in order of relevance. |

### LangGraph workflow (`langgraph_app.py`)

`RAGState` carries the URL, video ID, query, transcript, chunks, embeddings, context, answer and timestamps through these nodes:

```
START → fetch_video_id ─┬─(index exists)──► search_vector_store → generate_response → END
                        └─(new video)─────► fetch_video_data → create_chunks
                                            → generate_embeddings → build_vector_store
                                            → search_vector_store → …
```

The conditional edge (`check_vector_store`) skips ingestion for videos that are already indexed. Every node is decorated with `@traceable` and tagged with metadata (model names, chunking strategy) for the LangSmith project `YT_RAG_langgraph`.

---

## Project structure

```
YT-RAG/
├── app.py                      # Entry point: runs the LangGraph workflow on a URL + query
├── langgraph_app.py            # LangGraph StateGraph definition (rag_workflow)
├── run_eval_pipeline.py        # Entry point for the evaluation suites
├── src/yt_rag/
│   ├── components/
│   │   ├── data_loader.py      # Transcript fetch, preprocessing, semantic chunking
│   │   ├── embedding.py        # OpenAI embedding wrapper
│   │   ├── vectorstore.py      # FAISS store (per video) + VectorStoreManager.reset()
│   │   └── search.py           # RAGSearch: build/load index, retrieve, generate, timestamps
│   ├── evals/
│   │   ├── retriever_eval.py        # DeepEval: Contextual Recall / Precision
│   │   ├── generator_eval.py        # DeepEval: Faithfulness / Answer Relevancy (gold context)
│   │   ├── full_rag_eval_pipeline.py# DeepEval: end-to-end incl. Contextual Relevancy
│   │   ├── g_eval_deepeval.py       # DeepEval G-Eval: Correctness / Clarity with rubrics
│   │   └── eval_pipeline.py         # LangSmith evaluate(): custom LLM-as-judge + timestamp error
│   ├── prompts/
│   │   ├── system_prompt.txt        # Generation prompt
│   │   └── eval_prompts/            # Judge prompts: correctness, groundedness, relevance, retrieval relevance
│   ├── logger.py               # Timestamped file logging → logs/
│   ├── exceptions.py           # CustomException with file/line context
│   └── utils.py                # Pickle I/O, prompt loader, eval result saving, timestamp parsing
├── rag_evaluation/
│   ├── evaluation_set/         # Hand-written QA sets per video (question, answer, timestamp, difficulty)
│   │   └── deep_eval_goldens/  # DeepEval goldens (input, expected_output, gold context)
│   └── eval_results/           # CSV outputs from every eval run
├── notebooks/                  # Experiments: data loading, RAG search, LangGraph, eval analysis
├── faiss_store/<video_id>/     # Persisted indexes (generated)
├── data/                       # Latest chunks.pkl / embeddings.pkl (generated, intermediate)
├── logs/                       # Run logs (generated)
├── pyproject.toml / uv.lock    # uv-managed dependencies (Python ≥ 3.13)
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.13+, an OpenAI API key, and optionally a LangSmith account for tracing and evals.

```bash
git clone https://github.com/akshaypratap008/YouTube-RAG.git
cd YouTube-RAG

# with uv (recommended — uses uv.lock)
uv sync

# or with pip
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...                 # optional, for faster sentence-transformers downloads

# LangSmith tracing (optional, required for eval_pipeline.py)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=YouTube-RAG
```

> Run every command from the **project root**. Imports use the `src.yt_rag` prefix, and paths like `faiss_store/`, `data/` and `src/yt_rag/prompts` are relative to the working directory.

---

## Usage

### LangGraph workflow

```python
from langgraph_app import rag_workflow

state = {
    "video_url": "https://www.youtube.com/watch?v=98o_L3jlixw",
    "query": "Who is the host in the podcast?",
}
result = rag_workflow.invoke(state, config={"run_name": "langgraph_full_workflow"})

print(result["answer"])
print(result["time_stamps"])   # [(start_sec, end_sec), ...] in relevance order
```

Or edit the URL and query in `app.py` and run:

```bash
uv run python app.py
```

### Using `RAGSearch` directly

```python
from src.yt_rag.components.search import RAGSearch

rag = RAGSearch(url="https://www.youtube.com/watch?v=7ARBJQn6QkM")  # builds the index on first use
chunks = rag.search("How does the RTX 50 Series use AI to render images?", top_k=5)
answer = rag.generate_response(context=" ".join(chunks), query="How does the RTX 50 Series use AI to render images?")
print(answer, rag.get_video_timestamps())
```

To clear every stored index:

```python
from src.yt_rag.components.vectorstore import VectorStoreManager
VectorStoreManager().reset()
```

> Video IDs are extracted from `youtube.com/watch?v=<id>` URLs. `youtu.be/` short links and `/shorts/` URLs aren't supported yet. The video must have a transcript, either manual or auto-generated.

---

## Evaluation

Retrieval and generation are evaluated **separately** and then **end to end**, so a failure can be traced to the right stage. Evaluation data covers 5 videos (podcasts, a tech keynote, a data-science mock interview). Each QA pair has an answer timestamp and a difficulty label.

| Suite | File | What it measures | Judge |
|---|---|---|---|
| Retriever | `retriever_eval.py` | Contextual **Recall** (relevant chunks missed?) and **Precision** (how much noise?) | `gpt-4.1-mini` |
| Generator | `generator_eval.py` | **Faithfulness** and **Answer Relevancy**, using the *gold* context so retrieval is held constant | `gpt-4o-mini` |
| End to end | `full_rag_eval_pipeline.py` | Faithfulness, Answer Relevancy and **Contextual Relevancy** on retrieved context | `gpt-4o-mini` |
| G-Eval | `g_eval_deepeval.py` | Rubric-based **Correctness** (contradictions penalised, omissions not) and **Clarity** | configurable |
| LangSmith | `eval_pipeline.py` | Custom 1–10 judges for correctness, groundedness, relevance and retrieval relevance, plus **timestamp error** (seconds between the gold timestamp and the top chunk's start) | `gpt-4o-mini` |

Run a suite by uncommenting the matching block in `run_eval_pipeline.py`:

```bash
uv run python run_eval_pipeline.py
```

Results are appended to CSVs in `rag_evaluation/eval_results/`.

### Results so far

**Retriever ablation** (top_k = 5, max 300 words per chunk, mean across 5 eval sets):

| Embedding model | Chunk similarity threshold | Contextual Recall | Contextual Precision |
|---|---|---|---|
| text-embedding-3-large | 0.45 | **0.95** | **0.89** |
| text-embedding-3-large | 0.60 | 0.95 | 0.88 |
| text-embedding-3-small | 0.45 | 0.92 | 0.84 |

**Generation:**

| Suite | Metric | Mean score |
|---|---|---|
| Generator (gold context) | Faithfulness / Answer Relevancy | 0.85 / 0.73 |
| End to end | Faithfulness / Answer Relevancy / Contextual Relevancy | 0.83 / 0.79 / 0.22 |
| G-Eval (50 cases) | Correctness / Clarity | 0.93 / 0.93 |

The low contextual relevancy score is expected with large semantic chunks: each chunk holds the answer plus a lot of surrounding talk. Smaller chunks or a reranking step are the obvious next things to try.

---

## Tech stack

`youtube-transcript-api` · `sentence-transformers` (all-MiniLM-L6-v2) · OpenAI `text-embedding-3-large` / `gpt-4o-mini` · `faiss-cpu` · LangChain · LangGraph · LangSmith · DeepEval · pandas · uv


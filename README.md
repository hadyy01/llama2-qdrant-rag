# PDF RAG with LLaMA-cpp, FAISS & Qdrant

A Retrieval-Augmented Generation (RAG) pipeline that ingests **PDF documents**, builds a vector index, and answers questions using a local **LLaMA-2** model via `llama-cpp-python`.

## Architecture

```
PDF Files
   │
   ▼
PyPDFDirectoryLoader (LangChain)
   │
   ▼
RecursiveCharacterTextSplitter (chunk_size=10,000)
   │
   ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
   │
   ├──► FAISS (local vector store)
   └──► Qdrant Cloud (persistent vector store)
         │
         ▼
   RetrievalQA (LangChain)
         │
         ▼
   LlamaCpp (local LLaMA-2 inference)
         │
         ▼
   Answer
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your Qdrant and OpenAI credentials
```

Set environment variables:

```bash
export QDRANT_URL=https://your-cluster.qdrant.io:6333
export QDRANT_API_KEY=your_key
```

## Usage

1. Place your PDF files in `/content/` (or update `PyPDFDirectoryLoader` path)
2. Open `llama2_qdrant_rag.ipynb` and run all cells
3. Ask questions about your documents:

```python
question = "What are the key findings in the paper?"
answer = create_answer_with_context(question)
print(answer)
```

## Vector Stores

The notebook demonstrates both:
- **FAISS** — fast local indexing, no server needed
- **Qdrant Cloud** — persistent, production-ready vector database

## License

MIT

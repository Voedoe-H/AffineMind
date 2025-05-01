# AffineMind
**AffineMind** is a lightweight Retrieval-Augmented Generation (RAG) system designed to answer questions about formal methods and anomaly detection in cyber-physical systems. It leverages affine arithmetic, AACDDs, and symbolic modeling to provide explainable, research-grounded answers in real time.

## Features
- Embedding-based retrieval using SentenceTransformers
- Local LLM generation (TinyLlama or any GGUF-compatible model via `llama.cpp`)
- Context transparency: displays similarity scores with retrieved context chunks
- Real-time interactive UI with Gradio

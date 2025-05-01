# AffineMind
**AffineMind** is a lightweight Retrieval-Augmented Generation (RAG) system designed to answer questions about formal methods and anomaly detection in cyber-physical systems. The system focuses on the research regarding but not limited to the following papers:

- Reliable and Real-Time Anomaly Detection for Safety-Relevant Systems
  Hagen Heermann, Johannes Koch, Christoph Grimm
  Proceedings of DVCON 2024 (Accepted for presentation)
- Bridging the Gap Between Anomaly Detection and Runtime Verification: H-Classifiers
  Hagen Heermann, Christoph Grimm
  Proceedings of DATE 2025 (Accepted for presentation)


## Features
- Embedding-based retrieval using SentenceTransformers
- Local LLM generation (TinyLlama or any GGUF-compatible model via `llama.cpp`)
- Context transparency: displays similarity scores with retrieved context chunks
- Real-time interactive UI with Gradio

## Structure and How to Use

Install dependencies using the requirements.txt file. When cloned create a dir called models. In this dir save the "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"(https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) model file. With this done you can execute the main.py script to start the assistent. You can access the promp interface under "http://127.0.0.1:7860".

Also setup is a small test environment in the testRAGPipe.py script. This utilizes a set of ground truth QA samples in the testingqa.json. The script computes the awnser using the RAG setup from the main script. Then it computes the difference between the expected answer and the answer by the model. The hypothisis is that the better the whole RAG pipeline works the closer to the ground truth the responses will be once embedded.

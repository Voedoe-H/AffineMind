import gradio as gr
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
from llama_cpp import Llama
from transformers import AutoTokenizer

stTransformer = SentenceTransformer("all-MiniLM-L6-v2")
llm = Llama(
        model_path="models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )
production = True


with open("embeddings.json","r") as f:
    data = json.load(f)

texts = [pair["text"] for pair in data["Embedpairs"]]
vectors = np.array([pair["embedding"] for pair in data["Embedpairs"]], dtype="float32")
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

def retrieve_context(query, top_k=6):
    query_vec = stTransformer.encode(query).astype("float32")
    D, I = index.search(np.array([query_vec]), top_k)
    matches = [(texts[i], float(D[0][j])) for j, i in enumerate(I[0])]
    return matches

def build_prompt(context_chunks, query):
    context_texts = [chunk for chunk, _ in context_chunks]
    context = "\n---\n".join(context_texts)
    prompt = f"""You are a formal methods expert. Given the following context from technical documents, answer the user's question as clearly and precisely as possible.
    
    Only use information from the context below. If you cannot answer, say: "The answer is not in the provided context."

    Context:
    {context}
    
    Question:
    {query}
    
    Awnser:
    """
    return prompt
def answer_query(query):
    matches = retrieve_context(query)
    prompt = build_prompt(matches, query)
    result = llm(prompt, max_tokens=500,temperature=0.2)
    answer = result["choices"][0]["text"].strip()
    
    formatted_context = "\n\n".join(
        [f"Score: {distance:.4f}\n{text}" for text, distance in matches]
    )
    return answer, formatted_context


if __name__ == "__main__":
    if production:
        iface = gr.Interface(
            fn=answer_query,
            inputs=gr.Textbox(lines=3, placeholder="Ask a question about your research..."),
            outputs=[
                gr.Textbox(label="Answer"),
                gr.Textbox(label="Retrieved Contexts with Distances")
            ],
            title="Research QA Assistant",
            description="Ask about AACDDs, hybrid systems, anomaly detection, or your formal models."
        )

        iface.launch()

    else:
        pass
import gradio as gr
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
from llama_cpp import Llama
from transformers import AutoTokenizer

stTransformer = SentenceTransformer("all-MiniLM-L6-v2")



if __name__ == "__main__":
    with open("testingqa.json",'r') as f:
        dta = json.load(f)

    print(dta["testingQAPairs"])

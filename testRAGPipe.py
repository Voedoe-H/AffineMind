import gradio as gr
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
from llama_cpp import Llama
from transformers import AutoTokenizer
import main
import matplotlib.pyplot as plt

stTransformer = SentenceTransformer("all-MiniLM-L6-v2")

if __name__ == "__main__":
    with open("testingqa.json",'r') as f:
        dta = json.load(f)
    
    tests = []

    for pair in dta["testingQAPairs"]:
        awnserEncoding = stTransformer.encode(pair["Awnser"])
        tests.append({ "Question" : pair["Question"],
                        "AwnserEncoding" : awnserEncoding
                      })
    

    modelAwnserGroundTruthDiff = []

    for idx,testPair in enumerate(tests):
        modelAwnser, _ = main.answer_query(testPair["Question"])
        modelAwnserEncoded = stTransformer.encode(modelAwnser)
        awnserSimilartiy = cosine_similarity([modelAwnserEncoded],[testPair["AwnserEncoding"]])
        modelAwnserGroundTruthDiff.append(awnserSimilartiy[0][0])
        print(f"{idx} : {awnserSimilartiy}")
        print(f"Model Response:{modelAwnser}")

    
    indices = list(range(len(modelAwnserGroundTruthDiff)))
    plt.plot(indices,modelAwnserGroundTruthDiff)
    plt.show()
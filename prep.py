import json
from sentence_transformers import SentenceTransformer

def generateEmbeddings():
    with open('snippets.json','r') as snippetsJ:
        dta = json.load(snippetsJ)
        

    snippets = dta["QuestionAwnsers"] #dta["texts"] + 
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(snippets)

    ret = {"Embedpairs":[]}

    for text,embedding in zip(snippets,embeddings):
        ret["Embedpairs"].append({ "text":text, "embedding":embedding.tolist() } )

    with open('embeddings.json','w') as f:
        json.dump(ret,f,indent=2)

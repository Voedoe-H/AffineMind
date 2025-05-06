import json
from langchain_community.chat_models import ChatLlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import numpy as np

def load_data(filepath="snippets.json"):
    try:
        with open(filepath,"r",encoding="utf-8") as rdta:
            dta = json.load(rdta)
            documents = []
            
            for snippet in dta.get("texts", []):
                documents.append(Document(page_content=snippet))

            for qa_pair in dta.get("QuestionAwnsers", []):
                    if "?" in qa_pair:
                        question, answer = qa_pair.split("?", 1)
                        documents.append(Document(
                            page_content=f"Question: {question.strip()}?\nAnswer: {answer.strip()}"
                        ))
                    else:
                        print(f"Warning: No question mark in QA pair: {qa_pair}")
        
            return documents
        
    except FileNotFoundError:
        print(f"File{filepath}")
    
    return []

if __name__ == "__main__":

    documents = load_data()

    if documents:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embedding_model)

        model = ChatLlamaCpp(
                model_path="models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
                n_ctx=2048,
                n_threads=4,
                temperature=0.9,
                n_batch=64,
                n_gpu_layers=0,
                max_tokens=500,
                verbose=False
            )
        
        system_template = "You are a formal methods expert. Given the following context from technical documents, answer the user's question as clearly and precisely as possible.Only use information from the context below. If you cannot answer, say: The answer is not in the provided context."

        prompt_template = ChatPromptTemplate(
            [("system",system_template),("user","{text}")]
        )

        user_text = "What is a formal method?"

        prompt = prompt_template.invoke({"text":user_text})


        print(model.invoke(prompt).content)


  

    


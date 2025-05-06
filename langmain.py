import json
import gradio as gr
from langchain_community.chat_models import ChatLlamaCpp
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

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

def build_QA_Chain(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    model = ChatLlamaCpp(
            model_path="models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
            n_ctx=2048,
            n_threads=4,
            temperature=0.7,
            n_batch=64,
            n_gpu_layers=0,
            max_tokens=500,
            verbose=False
    )
        
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    return qa_chain


if __name__ == "__main__":

    documents = load_data()

    if documents:
        qa_chain = build_QA_Chain(documents)

        def awnser_query(query):
            result = qa_chain.invoke({"query": query})
            context = "\nRetrieved Docs:\n"
            for doc in result["source_documents"]:
                context += doc.page_content
            return result["result"], context

        iface = gr.Interface(
            fn=awnser_query,
            inputs=gr.Textbox(lines=3, placeholder="Ask a question about your research..."),
            outputs=[
                gr.Textbox(label="Answer"),
                gr.Textbox(label="Retrieved Contexts with Distances")
            ],
            title="Research QA Assistant",
            description="Ask about AACDDs, hybrid systems, anomaly detection, or your formal models."
        )

        iface.launch()
    


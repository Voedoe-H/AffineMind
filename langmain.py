import json
import gradio as gr
from langchain_community.chat_models import ChatLlamaCpp
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

def load_data(filepath="snippets.json"):
    """
        Function that trys to load the corpus which is expected to be in a json file with one list of texts saved under the key "texts". On the other hand there is a list
        of texts expected in the json file under the key "QuestionAwnsers" that contain strings of the form "Q?A" where Q is a question and A is an awnser.
        The output is a list of langchains document objects.
    """
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

def build_QA_Chain(documents,config):
    """

    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config["number_retrived_texts"]})

    model = ChatLlamaCpp(
            model_path="models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
            n_ctx = config["context_window_size"],
            n_threads = config["number_runners"],
            temperature= config["temperature"],
            n_batch=64,
            n_gpu_layers=0,
            max_tokens= config["number_output_tokens"],
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
    with open("config.json","r") as rconf:
        config = json.load(rconf)
    print(config)

    if documents and config :
        qa_chain = build_QA_Chain(documents,config)

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
    


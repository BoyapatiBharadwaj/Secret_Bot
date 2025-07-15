from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

def build_vector_store(docs, persist_path="vector_store/"):
    embeddings = SentenceTransformerEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_rag_chain(db):
    llm = Ollama(model="llama2")  # Llama2 must be running via Ollama
    retriever = db.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

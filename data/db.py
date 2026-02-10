from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import  HuggingFaceEmbeddings

def setup_vector_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter  = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap= 50 
    )
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name = 'sentence_transformer/all-mpnet-base-v2'
    )
    vector_db = FAISS.from_documents(
        chunks, embeddings
    )

def get_local_content(vector_db, query):
    docs = vector_db.similart_search(query, k=5)
    return ' '.join([doc.page_content for doc in docs])

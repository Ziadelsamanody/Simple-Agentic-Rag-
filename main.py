import os 
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from crewai.tools import SerperDevTool
from crewai import Agent, Task , Crew, LLM
from utils import check_local_knowledge
from dotenv import load_dotenv
load_dotenv()

Groq_API = os.environ.get('Groq_API')
Gemini_API = os.environ.get('Gemini_API')
Serper_API = os.environ.get('Serper_API')

llm = ChatGroq(
    model = 'llama-3.3-70b-specdec',
    temperature= 0,
    max_tokens= 500,
    timeout= None,
    max_retries= 2
)

crew_llm = LLM(
    model = 'gemini/gemini-1.5-flash',
    api_key=Gemini_API,
    max_tokens=500,
    temperature=0.7
)

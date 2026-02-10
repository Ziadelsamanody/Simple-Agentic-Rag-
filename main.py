import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai import  LLM
from dotenv import load_dotenv
from utils import check_local_knowledge, get_web_content , setup_web_scarapign_agent
from data import setup_vector_store, get_local_content
load_dotenv()
# knowledge = check_local_knowledge()
Groq_API = os.environ.get('Groq_API')
Gemini_API = os.environ.get('Gemini_API')
Serper_API = os.environ.get('Serper_API')

llm = ChatGroq(
    model = 'llama-3.3-70b-versatile',
    temperature= 0,
    max_tokens= 500,
    timeout= None,
    max_retries= 2,
    api_key= Groq_API
)

crew_llm = LLM(
    model = 'gemini/gemini-1.5-flash',
    api_key=Gemini_API,
    max_tokens=500,
    temperature=0.7
)

def get_final_answer(context, query):
    message = [
        ('system' , 'You are a helpful assistant. Use the provided context to answer the query accuratly'),
        ('system', f'Context: {context}'),
        ('human', query)
    ]
    response = llm.invoke(message)
    return response.content

def process_query(query, vector_db, local_context):
    # can_answer_locally = check_local_knowledge(query=query, context=local_context, llm=llm)
    # if can_answer_locally:
    #     context = get_local_content(vector_db, query)
    # else : 
    #     context = get_web_content(query, llm=crew_llm)
    # For now, only use local content
    context = get_local_content(vector_db, query)
    return get_final_answer(context, query)

def main(): 
    pdf_path = 'Bulding agent rag.pdf'
    vector_db = setup_vector_store(pdf_path)
    local_context = get_local_content(vector_db, '')
    query =  'What is Agentic Rag'
    result = process_query(query, vector_db, local_context)
    print(f'User : {query}')
    print(f"Bot : {result}")
if __name__ == '__main__':
    main()
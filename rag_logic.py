import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def build_rag_system(file_path):
    # Крок 1: Завантаження PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Крок 2: Нарізка тексту на шматки
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Крок 3: Створення векторної бази знань у папці 'db'
    persist_directory = 'db'
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
    
    # Крок 4: Налаштування логіки відповіді
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2) # Трохи більше творчості
    
    # Створюємо ретривер, який бере 7 найрелевантніших шматків тексту
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    
    prompt = ChatPromptTemplate.from_template("""
    You are a professional Python instructor. Answer the question using ONLY the provided context. 
    If you find code examples in the context, please include them in your answer and explain how they work.
    
    Context:
    {context}
    
    Question: {input}
    
    Answer:""")
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain
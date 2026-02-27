import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

def build_rag_system(file_path):
    # Step 1: Load and split the document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Step 2: Create Vector Store
    persist_directory = 'db'
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Step 3: Create Contextualize Prompt
    # This sub-chain reformulates the question based on chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Step 4: Create QA Chain
    system_prompt = (
        "You are a professional Python instructor. Answer the question using ONLY the provided context. "
        "If you find code examples in the context, please include them in your answer. "
        "If you don't know the answer, say that you don't know.\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Final assembly
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    
    return retrieval_chain
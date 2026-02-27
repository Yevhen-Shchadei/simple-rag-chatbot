import os
from rag_logic import build_rag_system
from langchain_core.messages import HumanMessage, AIMessage

PDF_FILE_PATH = "my_data.pdf"

def run_bot():
    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: File {PDF_FILE_PATH} not found!")
        return

    print("Status: Building the knowledge base...")
    qa_chain = build_rag_system(PDF_FILE_PATH)
    
    # This list will store our conversation
    chat_history = [] 
    
    print("Status: Ready! Type 'exit' to stop.")

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit']:
            break
            
        # Passing both user input and history to the chain
        result = qa_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        answer = result['answer']
        print(f"\nAI: {answer}")
        
        # Adding current turn to memory
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=answer),
        ])

if __name__ == "__main__":
    run_bot()
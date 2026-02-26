import os
from rag_logic import build_rag_system

# Шлях до твого PDF
PDF_FILE_PATH = "my_data.pdf"

def run_bot():
    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: File {PDF_FILE_PATH} not found!")
        return

    print("Status: Building the knowledge base...")
    
    # Ініціалізація системи
    qa_chain = build_rag_system(PDF_FILE_PATH)
    
    print("Status: Ready! Type 'exit' to stop.")

    # Цикл чату
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit']:
            break
            
        # Отримуємо відповідь. ВАЖЛИВО: передаємо словник {"input": ...}
        result = qa_chain.invoke({"input": user_input})
        
        # Виводимо результат за ключем 'answer'
        print(f"\nAI: {result['answer']}")

if __name__ == "__main__":
    run_bot()
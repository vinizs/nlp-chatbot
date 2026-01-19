from src.chatbot import NLPBot

def main():
    print("System: Initializing engines...")
    bot = NLPBot('data/questions.csv')
    
    print("\n" + "="*50)
    print("ADHD HYBRID BOT: ONLINE")
    print("I can handle greetings and technical ADHD questions.")
    print("Type 'exit' to quit.")
    print("="*50 + "\n")

    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue

        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("Bot: Goodbye! Wishing you a productive day.")
            break
        
        response = bot.get_response(user_input)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()
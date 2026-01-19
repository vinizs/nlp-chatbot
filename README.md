# Hybrid NLP & Markov ADHD Chatbot

An intelligent, natural-sounding chatbot built with **Python 3.9.6**. This assistant uses a hybrid approach to provide medically accurate ADHD information from a CSV knowledge base while maintaining a conversational, non-robotic tone.

## 🧠 The "Hybrid" Intelligence
Unlike standard FAQ bots that just copy-paste text, this engine uses three distinct layers:
1.  **Greeting Layer:** Instant response handling for "small talk" (Hi, Hello, Thanks).
2.  **NLP Retrieval (TF-IDF):** Mathematically finds the most accurate answer from a 1,000+ row CSV using Cosine Similarity.
3.  **Generative Synthesis (Markovify):** Uses Markov Chains trained on your data to generate unique "supporting" sentences, ensuring the bot doesn't sound repetitive.



## 📂 Project Structure
```text
nlp-chatbot/
├── data/
│   └── questions.csv     # Knowledge base (question_id, question, answer)
├── src/
│   ├── __init__.py
│   └── chatbot.py        # Hybrid NLP & Markovify logic
├── main.py               # User interface & terminal loop
├── requirements.txt      # Dependencies (scikit-learn, nltk, markovify)
├── .gitignore            # Keeps your repo clean
└── README.md             # You are here!
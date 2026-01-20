# 🧠 ADHD NLP Assistant (Vector-Search Edition)

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![NLP](https://img.shields.io/badge/NLP-Scikit--Learn-orange)

A high-precision AI assistant that uses **Mathematical Vector Space Modeling** to provide ADHD information. By converting text into numerical coordinates, the bot finds the most relevant answers from its database using geometry rather than simple keyword matching.

---

## ✨ Features
- **Deterministic Accuracy:** Uses TF-IDF (Term Frequency-Inverse Document Frequency) to weigh the importance of specific ADHD terms.
- **Cosine Similarity Engine:** Calculates the angle between user queries and stored data to find the "best fit" response.
- **Efficient Preprocessing:** Implements Lemmatization and Stop-word removal to filter out noise (like "the", "a", "is").
- **Clean Web Interface:** A minimalist chat UI built with Streamlit.



---

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Framework:** Streamlit
* **Analysis:** Scikit-Learn (TF-IDF Vectorizer)
* **Data Handling:** Pandas
* **Linguistics:** NLTK (WordNet Lemmatizer)

---

## 📁 Project Structure
```text
nlp-chatbot/
├── data/
│   └── questions.csv     # Knowledge Base
├── src/
│   └── chatbot.py        # NLP Engine (Vectorization & Similarity)
├── app.py                # Web Interface
├── requirements.txt      # Simplified Dependencies
└── README.md             # Documentation
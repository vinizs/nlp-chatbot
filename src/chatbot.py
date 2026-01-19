import pandas as pd
import re
import random
import nltk
import markovify
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup NLTK
nltk.download(['punkt', 'wordnet', 'stopwords'], quiet=True)

class NLPBot:
    def __init__(self, csv_path):
        try:
            self.df = pd.read_csv(csv_path)
            combined_text = " ".join(self.df['answer'].astype(str))
            self.markov_model = markovify.Text(combined_text)
        except Exception as e:
            print(f"Error initializing: {e}")
            self.df = pd.DataFrame(columns=['question', 'answer'])

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        
        if not self.df.empty:
            processed_questions = self.df['question'].apply(self.clean_text)
            self.question_vectors = self.vectorizer.fit_transform(processed_questions)

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        tokens = nltk.word_tokenize(text)
        cleaned = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words]
        return " ".join(cleaned)

    def check_greetings(self, user_input):
        """Returns a response if the input is a common greeting."""
        user_input = user_input.lower().strip()
        greetings = {
            "hi": ["Hello! How can I help you with ADHD information today?", "Hi there! What's on your mind?"],
            "hello": ["Greetings! I'm here to answer your ADHD questions.", "Hello! Need some help?"],
            "hey": ["Hey! How can I assist you?", "Hi! Ready to learn about ADHD?"],
            "thanks": ["You're welcome!", "Happy to help!", "Anytime!"],
            "thank you": ["No problem at all.", "Glad I could assist!"]
        }
        for key in greetings:
            if user_input == key or user_input.startswith(key + " "):
                return random.choice(greetings[key])
        return None

    def generate_natural_bridge(self, raw_answer):
        generated = self.markov_model.make_short_sentence(100)
        openers = [
            f"Based on my data, {raw_answer}",
            f"Sure! {raw_answer}",
            f"Here is some information: {raw_answer}",
            raw_answer
        ]
        base_response = random.choice(openers)
        if generated and len(generated) > 15:
            # We lowercase the generated part to make it flow like a continuous thought
            return f"{base_response} Also, remember that {generated[0].lower() + generated[1:]}"
        return base_response

    def get_response(self, user_query):
        # 1. First, check if it's a greeting
        greeting = self.check_greetings(user_query)
        if greeting:
            return greeting

        # 2. If not a greeting, proceed to NLP search
        cleaned_query = self.clean_text(user_query)
        if not cleaned_query:
            return "I'm listening! Please ask me a question about ADHD."

        query_vec = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vec, self.question_vectors)
        best_match_idx = similarities.argmax()
        
        if similarities[0][best_match_idx] < 0.25:
            return "I'm not quite sure I have that specific information. Could you try rephrasing your question?"

        raw_answer = self.df.iloc[best_match_idx]['answer']
        return self.generate_natural_bridge(raw_answer)
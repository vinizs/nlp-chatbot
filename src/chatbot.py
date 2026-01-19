import pandas as pd
import re
import random
import nltk
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
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.df = pd.DataFrame(columns=['question', 'answer'])

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        
        if not self.df.empty:
            processed_questions = self.df['question'].apply(self.clean_text)
            self.question_vectors = self.vectorizer.fit_transform(processed_questions)

    def clean_text(self, text):
        """Cleans and normalizes text for better matching."""
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        tokens = nltk.word_tokenize(text)
        cleaned = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words]
        return " ".join(cleaned)

    def check_greetings(self, user_input):
        """Quick response for small talk."""
        user_input = user_input.lower().strip()
        greetings = {
            "hi": ["Hello! How can I help you today?", "Hi there! Ask me anything about ADHD."],
            "hello": ["Greetings! What would you like to know about ADHD?", "Hello!"],
            "thanks": ["You're very welcome!", "Happy to help!", "Anytime!"],
            "thank you": ["Glad I could assist!", "No problem at all."]
        }
        for key in greetings:
            if user_input == key or user_input.startswith(key + " "):
                return random.choice(greetings[key])
        return None

    def synthesize_response(self, user_query, raw_answer):
        """Analyzes intent and wraps the answer naturally."""
        query = user_query.lower()
        
        # 1. Detect Intent for the prefix
        if any(word in query for word in ["why", "reason", "cause"]):
            prefixes = ["That's because ", "The main reason is that ", "Essentially, "]
        elif any(word in query for word in ["how", "process", "way"]):
            prefixes = ["Usually, it works like this: ", "In many cases, ", "The process involves "]
        elif any(word in query for word in ["is it", "can i", "do i", "does"]):
            prefixes = ["Based on what I know, ", "Actually, ", "In short, "]
        else:
            prefixes = ["Here is some info: ", "Sure, ", "", "Interestingly, "]

        prefix = random.choice(prefixes)
        
        # 2. Format the raw answer to follow the prefix
        # If the answer is short (like "No."), don't lowercase it too much
        formatted_answer = raw_answer
        if prefix and len(raw_answer) > 0:
            # Lowercase the first letter of the answer if it follows a prefix
            formatted_answer = raw_answer[0].lower() + raw_answer[1:]

        return f"{prefix}{formatted_answer}"

    def get_response(self, user_query):
        # First check greetings
        greeting = self.check_greetings(user_query)
        if greeting:
            return greeting

        cleaned_query = self.clean_text(user_query)
        if not cleaned_query:
            return "I'm listening! Please ask me a question about ADHD."

        query_vec = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vec, self.question_vectors)
        best_match_idx = similarities.argmax()
        
        # Confidence threshold (0.25 is usually a safe bet)
        if similarities[0][best_match_idx] < 0.25:
            return "I'm not quite sure I have the answer to that specific question. Could you try rephrasing?"

        raw_answer = self.df.iloc[best_match_idx]['answer']
        
        # Use our new synthesis engine instead of the old Markov one
        return self.synthesize_response(user_query, raw_answer)
import openai
import json
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import dotenv
import os

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Chatbot:
    def __init__(self, system_prompt, agent_name):
        self.gpt = "gpt-3.5-turbo"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rake = Rake()
        self.sia = SentimentIntensityAnalyzer()
        self.system_prompt = system_prompt
        self.agent_name = agent_name
        self.stop = "/n"
        self.messages = [{"role": "system", "content": self.system_prompt}]
        
        self.memory_file = f'{self.agent_name}.json'
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                self.memory = json.load(f)
        else:
            self.memory = []
        
        self.sentiment_drift = np.zeros(4)
    
    def calculate_temperature(self):
        return 0.2 + 1.3 * (1 - self.sentiment_drift[1])
    
    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f)

    def chat(self, text, search_results=None):
        similar_response = self.search(text)

        if search_results is not None:
            text += ' ' + search_results

        if similar_response:
            self.messages.append({
                "role": "assistant",
                "content": f"Knowing what I do already about {text}, {similar_response}"
            })

        self.messages.append({"role": "user", "content": text})
        
        response = openai.ChatCompletion.create(
            model=self.gpt,
            messages=self.messages,
            temperature=self.calculate_temperature()
        )

        response_text = response['choices'][0]['message']['content']
        self.messages.append({"role": "assistant", "content": response_text})

        input_embedding = self.model.encode(text)

        self.rake.extract_keywords_from_text(text)
        keywords = self.rake.get_ranked_phrases()
        keyword_sentiments = {keyword: TextBlob(keyword).sentiment.polarity for keyword in keywords}
        input_sentiment = self.sia.polarity_scores(text)
        response_sentiment = self.sia.polarity_scores(response_text)
        
        self.sentiment_drift[0] += abs(input_sentiment['compound'])
        self.sentiment_drift[1] += input_sentiment['compound']
        self.sentiment_drift[2] += max(keyword_sentiments.values(), default=0)
        self.sentiment_drift[3] += 1
        self.sentiment_drift = self.sentiment_drift / np.linalg.norm(self.sentiment_drift)

        self.memory.append({
            'query': text,
            'response': response_text,
            'embedding': input_embedding.tolist(),
            'keyword_sentiments': keyword_sentiments,
            'input_sentiment': input_sentiment,
            'response_sentiment': response_sentiment
        })

        self.save_memory()

        return response_text

    def search(self, query):
        query_embedding = self.model.encode(query)
        if not self.memory:
            return " "
        similarities = [cosine_similarity([query_embedding], [np.array(interaction['embedding'])]) for interaction in self.memory]
        most_similar_index = np.argmax(similarities)
        most_similar_interaction = self.memory[most_similar_index]
        return most_similar_interaction['response']

chatbot = Chatbot("You are chatting with an AI assistant. ", "AI_Assistant")

user_input = input("Enter your message: ")
while user_input.lower() != 'quit':
    if user_input.lower().startswith('search '):
        search_query = user_input[len('search '):]
        search_results = chatbot.search(search_query)
        print(search_results if search_results else "No results found.")
    else:
        response = chatbot.chat(user_input)
        print(response)
    user_input = input("Enter your message: ")

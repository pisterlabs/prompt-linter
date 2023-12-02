import openai
import os
import pinecone
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load your OpenAI API key from an environment variable or secret management service
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Pinecone
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'))

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a vector index with Pinecone
index_name = "sentiment-index"
pinecone.deindex(index_name)
pinecone.create_index(index_name)

# Define a few example sentiments and add them to the index
sentiments = ["positive", "neutral", "negative"]
vectors = model.encode(sentiments)
pinecone.upsert(index_name=index_name, ids=sentiments, vectors=vectors)

class Journal:
    def __init__(self):
        self.entries = []

    def new_entry(self, json_input):
        # Parse the JSON input
        data = json.loads(json_input)
        text = data.get("text")
        mood = data.get("mood")
        
        self.entries.append({"text": text, "mood": mood})
        
        sentiment = self.analyze_sentiment(text)
        prompt = self.generate_prompt(sentiment, mood)
        question = self.generate_question(prompt)
        print(question)

    def analyze_sentiment(self, text):
        # Encode the text as a vector
        vector = model.encode([text])[0]
        
        # Use Pinecone to find the closest sentiment
        results = pinecone.query(index_name=index_name, queries=[vector], top_k=1)
        sentiment = results.ids[0][0]
        
        return sentiment

    def generate_prompt(self, sentiment, mood):
        # Generate a prompt based on the sentiment and user's selected mood
        prompt_base = "It seems like you're feeling {}. ".format(mood)
        
        if sentiment == "negative":
            return prompt_base + "Can you tell me more about what made your day tough?"
        elif sentiment == "neutral":
            return prompt_base + "What were the highs and lows of your day?"
        else:  # sentiment == "positive"
            return prompt_base + "That's great! Can you share more about what made your day fantastic?"

    def generate_question(self, prompt):
        # Generate a question using GPT-3
        response = openai.Completion.create(
          engine="text-davinci-003",
          prompt=prompt,
          temperature=0.5,
          max_tokens=60
        )
        return response.choices[0].text.strip()

journal = Journal()
journal.new_entry('{"text": "I had a great day!", "mood": "happy"}')

# Deindex and stop Pinecone when finished to avoid unnecessary costs
pinecone.deindex(index_name)
pinecone.deinit()

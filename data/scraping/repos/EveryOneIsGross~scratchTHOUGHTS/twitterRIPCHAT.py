'''
HERE IS THE INJEST AND CSV CREATING SCRIPT, JUST SUPPLY THE PATH TO YR DOWNLOADED.ZIP

import os
import zipfile
import json
import pandas as pd
from textblob import TextBlob

def get_sentiment(text):
    """Determine the sentiment of a given text."""
    analysis = TextBlob(text)
    # Classifying the polarity of the text
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

def process_twitter_archive(zip_file_path, output_folder_path):
    # Extract the archive
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder_path)

    # Read the contents of the 'tweets.js' file
    tweets_file_path = os.path.join(output_folder_path, "data", "tweets.js")

    with open(tweets_file_path, 'r', encoding='utf-8') as file:
        tweets_data = file.read()

    # Convert the content to valid JSON format
    tweets_json_str = tweets_data.replace('window.YTD.tweets.part0 = ', '')
    tweets_json = json.loads(tweets_json_str)
    tweets_list = [tweet['tweet'] for tweet in tweets_json]

    # Extract details and sentiments from each tweet
    tweet_texts = [tweet['full_text'] for tweet in tweets_list]
    sentiments = [get_sentiment(tweet) for tweet in tweet_texts]

    # Creating a DataFrame to structure the data
    tweets_df = pd.DataFrame({
        'Date': [tweet['created_at'] for tweet in tweets_list],
        'Tweet': tweet_texts,
        'Sentiment': sentiments
    })

    # Save the DataFrame to a CSV file
    csv_path = os.path.join(output_folder_path, "twitter_data_processed.csv")
    tweets_df.to_csv(csv_path, index=False)

    print(f"Processed data saved to: {csv_path}")

if __name__ == "__main__":
    input_zip_path = input("Please provide the path to your Twitter archive .zip file: ").strip()
    output_directory = input("Where would you like to save the processed CSV? (e.g., C:\\path_to_folder): ").strip()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    process_twitter_archive(input_zip_path, output_directory)

'''
import openai
from gpt4all import GPT4All, Embed4All
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import zipfile
import os
import pickle


openai.api_base = "http://localhost:4892/v1"
openai.api_key = "null"

# Define constants
model = "mistral trismegistus"
OPENAI_ENGINE = "model"

class ConversationalAgent:
    def __init__(self, csv_path):
        # Initialize the embedder
        self.embedder = Embed4All()
        
        # Load the CSV data
        self.data = pd.read_csv(csv_path)
        
        # Define the pickle file name
        pickle_filename = csv_path.split('\\')[-1].replace('.csv', '_embeddings.pkl')
        
        if os.path.exists(pickle_filename):
            with open(pickle_filename, 'rb') as f:
                self.tweet_embeddings = pickle.load(f)
        else:
            # Embed the tweets
            embeddings = [self.embedder.embed(tweet) for tweet in self.data['Tweet']]
            self.tweet_embeddings = np.vstack(embeddings)
            
            # Save the embeddings to the pickle file
            with open(pickle_filename, 'wb') as f:
                pickle.dump(self.tweet_embeddings, f)

    def find_most_similar_tweet(self, input_text):
        # Embed the user input
        input_embedding = np.array(self.embedder.embed(input_text)).reshape(1, -1)
        
        # Print shapes for debugging
        print("Shape of input_embedding:", input_embedding.shape)
        print("Shape of self.tweet_embeddings:", self.tweet_embeddings.shape)
        
        # Calculate cosine similarity between input and tweet embeddings
        similarities = cosine_similarity(input_embedding, self.tweet_embeddings)
        # Get the index of the most similar tweet
        most_similar_idx = np.argmax(similarities)
        return self.data.iloc[most_similar_idx]

    def find_top_similar_tweets(self, query, top_n=5):
        # Embed the query
        query_embedding = np.array(self.embedder.embed(query)).reshape(1, -1)
        # Calculate cosine similarity between the query and tweet embeddings
        similarities = cosine_similarity(query_embedding, self.tweet_embeddings)
        # Get the indices of the top n most similar tweets
        top_indices = similarities[0].argsort()[-top_n:][::-1]
        return self.data.iloc[top_indices]


        
    def generate_response(self, user_input):
        # Find the most similar tweet to the user's input
        similar_tweet = self.find_most_similar_tweet(user_input)
        # Use the sentiment, historical data, and tweet content to craft a response prompt
        prompt = (f"Based on a similar tweet from {similar_tweet['Date']}, "
                f"which had a '{similar_tweet['Sentiment']}' sentiment "
                f"and said \"{similar_tweet['Tweet']}\", I'd say: ")
        # Use OpenAI to generate a continuation of the response
        response = openai.Completion.create(model=model, prompt=prompt, max_tokens=100)
        return prompt + response.choices[0].text
    
    def chat(self):
        print("Chat with the agent! (Type 'exit' to stop)")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Agent: Goodbye!")
                break
            elif user_input.lower().startswith('search'):
                query = user_input[7:]  # Remove the 'search ' part
                similar_tweets = self.find_top_similar_tweets(query)
                print("\nTop similar tweets based on your query:")
                for idx, row in similar_tweets.iterrows():
                    print(f"- {row['Sentiment']}: {row['Tweet']}\n")
            else:
                response = self.generate_response(user_input)
                print("Agent:", response)

# Initialize the agent
csv_path = "twitter_data_processed.csv"

agent = ConversationalAgent(csv_path)
agent.chat()

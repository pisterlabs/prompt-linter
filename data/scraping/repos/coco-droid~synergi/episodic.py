import pinecone
import openai
import time 
class EpisodicMemory:
    def __init__(self,index):
        self.pinecone_api_key = "8d5389f5-d30a-424b-a780-393ca5bcccfb"
        self.openai_api_key = "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"
        self.pinecone_index = "episodic"
        self.memory = []
        
        # Initialize Pinecone client
        pinecone.init(api_key=self.pinecone_api_key,
         environment='gcp-starter' 
        )
        self.index=index = pinecone.Index('episodic')
        # Set OpenAI API key
        openai.api_key = self.openai_api_key
        
    def store_message(self, user_message):
        text_encoded_message = self.get_embedding(user_message)
        vector_data = [{'id': user_message, 'values': text_encoded_message}]
        self.index.upsert(vectors=vector_data)
        
    def search_memory(self, query):
        query_embedding = self.get_embedding(query)
        results = self.index.query(query_embedding, top_k=5)['matches']
        results = [result.id for result in results]
        print(f"response to:{query} is:{results}")
        #loop through results and decode them
        for i in range(len(results)):
            results[i]=self.decode(results[i])
            print(results[i])
        return results
        
    def get_embedding(self, text):
        print(f"Generating embedding for: {text}")
        response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
        print(f"embed:{response['data'][0]['embedding']}")
        return response['data'][0]['embedding']
        
    def process_user_input(self, user_input):
        self.store_message(user_input)
        search_query = self.generate_search_query(user_input)
        search_results = self.search_memory(search_query)
        # Apply logic to prioritize frequently accessed data
        
        return search_results
    
    def generate_search_query(self, user_input):
        # Implement logic to extract keywords from user input for search
        keywords = self.extract_keywords(user_input)
        return " ".join(keywords)
    
    def extract_keywords(self, text):
        # Implement logic to extract keywords from user input
        # This could involve natural language processing techniques
        
        return keywords
    def decode (self,encoded_message):
        response = openai.Completion.create(model="text-embedding-ada-002", prompt=encoded_message)
        return response.choices[0].text
    def close(self):
        pinecone.deinit()

# Replace with your actual Pinecone and OpenAI API keys



# Close the agent
#agent.close()

from dotenv import load_dotenv
load_dotenv()
import openai

response = openai.Embedding.create(
    input="How many planets are there in the solar system?",
    model="text-embedding-ada-002"
)
embeddings = response['data'][0]['embedding']
print(embeddings)
print('-------')
print(len(embeddings))
import os
import openai
import pinecone
from dotenv import load_dotenv

load_dotenv()

# Set up the OpenAI API client
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "text-embedding-ada-002"

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="northamerica-northeast1-gcp"  # find next to API key in console
)

# Retrieve the chunks from Pinecone
pinecone_index = pinecone.Index(index_name="openai")
query = "What is the revenue of 'Family of Apps' division of Meta in 2022?"

# create the query embedding
xq = openai.Embedding.create(input=query, engine="text-embedding-ada-002")['data'][0]['embedding']

pinecone_results = pinecone_index.query(queries=[xq], top_k=3, include_metadata=True, namespace="meta")
chunks = [result['metadata']['text'] for result in pinecone_results['results'][0]['matches']]

# Concatenate the chunks into a single text
text = " ".join(chunks)
text += ". Answer the following question from the previous information: " + query

# Generate a response from OpenAI's language model
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=text,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0,
)

# Print the response
print(response.choices[0].text)

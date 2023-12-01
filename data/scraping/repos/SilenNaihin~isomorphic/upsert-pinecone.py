import os
import random
import csv
import openai
import pinecone
# List to hold dictionaries
poems = []
openai.api_key = os.environ["OPENAI_API_KEY"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_env = "asia-southeast1-gcp-free"
# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# Name of your existing index
index_name = "isomorphic"

# Initialize the Pinecone indexer
index = pinecone.Index(index_name=index_name)


with open('poems.csv', 'r') as file:
    # Create a csv reader
    reader = csv.reader(file)
    # Get the headers
    headers = next(reader)
    n=0
    for row in reader:
        if n > 50:
            break
        # Create a dictionary for each row and append it to the list
        poem = {headers[i]: value for i, value in enumerate(row)}
        poems.append(poem)
        n+=1
# Now 'poems' is a list of dictionaries where each dictionary has 'author',

# print(poems[0]['author'] + "\n")
# print(poems[0]['poem name'] + "\n")
# print("\n")
# print(poems[0]['content'])
print(f"poem num: {len(poems)}")
n=0
embeddings = []
for poem in poems:
    if n > 50:
        break
    content = poem['content']
     
    response = openai.Embedding.create(input=content, model="text-embedding-ada-002")
    embedding = response['data'][0]['embedding']
    embeddings.append(embedding)
    n+=1
print(f"embedding num: {len(embeddings)}")

vec_list = []
for embedding, poem in zip(embeddings, poems):
    metadata = {"author": str(poem["author"]), "title": str(poem["poem name"])}
    first_digit = str(random.randint(1, 9))
    rest_of_digits = ''.join(str(random.randint(0, 9)) for _ in range(15))
    id = str(int(first_digit + rest_of_digits))
    vec_list.append((id, embedding, metadata))
print(f"vec num: {len(vec_list)}")
# Execute the upsert operation
index.upsert(vec_list)

    

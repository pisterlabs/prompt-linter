import pandas as pd
from openai import OpenAI
from embed import get_embedding
import numpy as np

client = OpenAI()


def get_prompt_from_docs(query, docs):
    prompt = ""
    for doc in docs:
        prompt += "```\n"
        prompt += doc 
        prompt += "\n```\n"
    prompt += "Question: " + query + "\nAnswer:"
    return prompt

def generate_response(prompt):
    simple_system_prompt = open("prompts/simple_prompt.txt", "r").read()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": simple_system_prompt},
        {"role": "user", "content": prompt}
    ]
    )
    return completion.choices[0].message.content  

def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def main():
    ## read in the csv file
    df = pd.read_csv('notion_embeddings.csv')
    df['Embedding'] = df['Embedding'].apply(lambda x: np.fromstring(x, sep=','))
    NUM_DOCS = 4
    COSINE_THRESHOLD = 0.1
    ## get input from the user for a query
    while True:
        query = input("Ask a question about yourself: ")
        query_embedding = get_embedding(query)
        ## get the embeddings from the csv file and calculate the cosine similarity
        df['cosine_similarity'] = df['Embedding'].apply(lambda x: cosine_similarity(x, query_embedding))

        ## sort the dataframe by cosine similarity
        df = df.sort_values(by=['cosine_similarity'], ascending=False)
        top_docs = df.head(NUM_DOCS)
        top_docs = top_docs[top_docs['cosine_similarity'] > COSINE_THRESHOLD]

        ## get the prompt from the top docs
        prompt = get_prompt_from_docs(query, top_docs['Page Text'].tolist())

        print("======= GENERATED PROMPT: ")
        print(prompt)

        print("======== GENERATED RESPONSE: ")
        resp = generate_response(prompt)

        print(resp)

if __name__ == "__main__":
    main()
import openai
import pinecone
import os

limit = 3750

# connect to pinecone_index
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment="northamerica-northeast1-gcp" # next to api key in console
)
index_name = "nptel"
pinecone_index = pinecone.Index(index_name=index_name)
embed_model = "text-embedding-ada-002"

def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = pinecone_index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    # TODO: Write a logging function that helps log the question and answer to mixpanel
    return [text_to_dict(course) for course in contexts]


def text_to_dict(text):
    # Split the text on newline characters
    lines = text.split("\n")
    
    # Initialize an empty dictionary
    result = {}
    
    # Iterate through the lines
    for line in lines:
        # Split the line into key and value using the first colon as a delimiter
        key, value = line.split(":", 1)
        
        # Strip any leading or trailing spaces from the key and value
        key = key.strip()
        value = value.strip()
        
        # Add the key-value pair to the result dictionary
        result[key] = value

    return result

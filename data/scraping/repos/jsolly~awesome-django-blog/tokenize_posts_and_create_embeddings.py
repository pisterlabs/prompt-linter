import tiktoken
import pandas as pd
from pathlib import Path
import json
import openai

# import matplotlib.pyplot as plt

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Three levels up

# Load the JSON file as a list of dictionaries
with open(
    BASE_DIR / "utilities/create_embeddings/processed/processed_posts.json",
    "r",
) as f:
    data = json.load(f)

# Create a dataframe from the list of dictionaries
df = pd.DataFrame(
    data, columns=["title", "category", "date_posted", "author", "content"]
)

# Tokenize the text and save the number of tokens to a new column
df["n_tokens"] = df.content.apply(lambda x: len(tokenizer.encode(x)))

max_tokens = 500


# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens=max_tokens):
    # Split the text into sentences
    sentences = text.split(". ")

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):
        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


shortened = []

# Loop through the dataframe
for row in df.iterrows():
    # If the content is None, go to the next row
    if row[1]["content"] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]["n_tokens"] > max_tokens:
        shortened += split_into_many(row[1]["content"])

    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append(row[1]["content"])

# Create a new dataframe from the list of shortened texts
df_shortened = pd.DataFrame(shortened, columns=["content"])

# Visualize the distribution of the number of tokens per row using a histogram
df_shortened["n_tokens"] = df_shortened.content.apply(
    lambda x: len(tokenizer.encode(x))
)

# Show the plot
# df_shortened.n_tokens.hist()
# plt.show()

# Use the OpenAI Embedding API to generate embeddings for each text
embeddings = []

for row in df_shortened.iterrows():
    response = openai.Embedding.create(
        input=row[1]["content"], engine="text-embedding-ada-002"
    )
    embeddings.append(response["data"][0]["embedding"])

# Add the embeddings as a new column to the dataframe
df_shortened["embeddings"] = embeddings

# Save the dataframe to a JSON file in the processed directory
json_path = BASE_DIR / "utilities/create_embeddings/processed/embeddings.json"
df_shortened.to_json(json_path, orient="records")

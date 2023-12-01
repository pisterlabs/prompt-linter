import sys
import pandas as pd
import json
from sklearn.manifold import TSNE
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# Initialize the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to calculate the embedding for a text string
def calculate_embedding(text):
    # Perform the embedding calculation for the text string
    embedding = embeddings.embed_query(text)
    return embedding

# Get the input file path from command-line argument
input_file_path = sys.argv[1]

# Generate the output file path by appending "-output" to the input file name
output_file_path = input_file_path.replace(".csv", "-output.csv")

# Read the input CSV file using pandas
data = pd.read_csv(input_file_path, header=None, skiprows=1)

# Extract issue keys and descriptions from the DataFrame
issue_keys = data.iloc[:, 0].values
descriptions = data.iloc[:, 1].values

# Calculate embeddings for each description
embeddings = [calculate_embedding(text) for text in descriptions]

# Convert the embeddings list to a numpy array
embeddings_array = np.array(embeddings)

# Apply t-SNE for dimension reduction
tsne = TSNE(n_components=2)
reduced_data = tsne.fit_transform(embeddings_array)

# Create a new DataFrame with the issue key and reduced coordinates
output_data = pd.DataFrame({
    "issue-key": issue_keys,
    "x-coordinate": reduced_data[:, 0],
    "y-coordinate": reduced_data[:, 1]
})

# Save the output DataFrame to a new CSV file
output_data.to_csv(output_file_path, index=False)

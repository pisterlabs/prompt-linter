import pandas as pd
import openai
import os
from pathlib import Path
from dotenv import load_dotenv
from openai.embeddings_utils import get_embedding
import csv

env_path = Path('../..') / '.env'
if env_path.exists():
    load_dotenv()

# Retrieve environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

df = pd.read_csv('data/output.csv')

# Prepare the CSV writer
csv_file = open('embeddings/openai-ada/all-command_embedding.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(df.columns.tolist() + ['embedding'])

# Apply the get_embedding function to each description
for i, row in df.iterrows():
    embedding = get_embedding(row['description'], engine='text-embedding-ada-002')
    # Write the row with its new 'embedding' column to the CSV file
    csv_writer.writerow(row.tolist() + [embedding])
    if i % 100 == 0:  # Print a message every 100 rows
        print(f"Processed row: {i}")

csv_file.close()



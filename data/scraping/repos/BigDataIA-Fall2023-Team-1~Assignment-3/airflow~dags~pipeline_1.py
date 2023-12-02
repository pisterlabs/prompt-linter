import pandas as pd
import openai
import os
import boto3
from io import StringIO
from dotenv import load_dotenv 

load_dotenv()

# Set your OpenAI API key here
api_key = os.getenv('AIRFLOW_VAR_OPENAI_API_KEY')

# Set your S3 credentials
aws_access_key_id = os.getenv('AIRFLOW_VAR_AWS_ACCESS_KEY')
aws_secret_access_key = os.getenv('AIRFLOW_VAR_AWS_SECRET_KEY')
s3_bucket_name = os.getenv('AIRFLOW_VAR_S3_BUCKET_NAME')

openai.api_key = api_key

# Initialize S3 client
s3 = boto3.client('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key)

# Load the CSV file containing form information
def read_csv_file():
    # Get the directory of the DAG file
    current_directory = os.path.dirname(os.path.abspath(__file__ or '.'))
    csv_file_path = os.path.join(current_directory, 'cleaned_file.csv')

    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    return df

def generate_embeddings(text_content):
    # Define your text-embedding model
    embedding_model = "text-embedding-ada-002"

    # Generate embeddings using OpenAI API
    response = openai.Embedding.create(model=embedding_model, input=text_content)

    return response['data'][0]['embedding']

def save_embeddings_to_s3(embeddings, s3_bucket_name):
    # Create a DataFrame with the embeddings
    df = pd.DataFrame({'embeddings': embeddings})

    # Convert DataFrame to CSV in memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    # Upload the CSV file to S3
    s3.upload_fileobj(csv_buffer, s3_bucket_name, 'embeddings.csv')

if __name__ == "__main__":
    df = read_csv_file()

    # Generate embeddings for all rows in the DataFrame
    embeddings_pypdf = df['pypdf_content'].apply(lambda x: generate_embeddings(x) if x else None)

    # Drop rows where embeddings could not be generated (e.g., due to empty content)
    embeddings_pypdf = embeddings_pypdf.dropna()

    # Check if there are any rows left after the drop
    if not embeddings_pypdf.empty:
        # Create a DataFrame with the embeddings
        df_pypdf = pd.DataFrame({'embeddings': embeddings_pypdf})

        df_pypdf.rename(columns={'embeddings': 'embeddings'}, inplace=True)

        df_final = pd.concat([df['pypdf_content'], df_pypdf], axis=1)

        # Convert DataFrame to CSV content as bytes
        csv_content = df_final.to_csv(index=False).encode('utf-8')

        # Save the embeddings for 'pypdf_content' to S3
        s3.put_object(Bucket=s3_bucket_name, Key='pypdf_embeddings.csv', Body=csv_content)
    else:
        print("No valid embeddings for 'pypdf_content' to save.")
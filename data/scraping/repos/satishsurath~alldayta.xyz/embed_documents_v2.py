# This code embeds all the text chunks from ChopDocuments.py
# Run this before you run app.py
# Look at "-originaltext.csv" before you run this to make sure you docs scanned right!
# You need an OpenAI key saved in APIkey.txt
# Note there is a limit to how many things you can embed with the OpenAI API, so I split the documents to stay under
# This will only add new documents; every time you add something new, Chop it with ChopDocuments, then run this

import os
import time
import pandas as pd
import numpy as np
import openai
from app import app
from app.routes_helper import retry_with_exponential_backoff  # Import the retry decorator with exponential backoff

# Global Variables
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_TOKENS_PER_BATCH = 250000  # Maximum tokens to be sent in a single batch to OpenAI API
WAIT_SECONDS = 10  # Wait time in seconds before sending next batch to OpenAI API

# Function to read settings from a file and return as a dictionary
def read_settings(file_name):
    with open(file_name, "r") as f:
        return {key: value for key, value in (line.strip().split("=") for line in f)}

# Function to get embeddings from OpenAI API
def embed_input_text(input_text_batch):
    try:
        return openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=input_text_batch
        )["data"]
    except openai.error.InvalidRequestError as e:
        print(f"Error with input: {input_text_batch}")
        raise e

# Main function to embed text and save as .npy files for each course
def embed_documents_given_course_name(course_name):
    app.logger.info(f"Embedding documents for course: {course_name}")
    MAX_TOKENS_PER_BATCH = 250000  # Max tokens per batch
    filedirectory = course_name  # Define file directory
    output_folder = os.path.join(course_name, "EmbeddedText")  # Define output folder

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # if os.getenv("OPENAI_API_KEY") is None then log an error and return
    if os.getenv("OPENAI_API_KEY") is None:
        app.logger.error("OPENAI_API_KEY is not set. Please set it in the environment variables.")
        return
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")  # Set OpenAI API key
        app.logger.info("OPENAI_API_KEY is set successfully.")
    
    folder = os.path.join(course_name, "Textchunks")  # Define the folder to read CSVs from

    # Loop through CSV files in the folder
    for file in os.listdir(folder):
        if file.endswith(".csv") and not file.startswith('CourseContentActivations'):
            filename_without_extension = os.path.splitext(file)[0]
            npy_filename = f"{filename_without_extension}.npy"
            output_path = os.path.join(output_folder, npy_filename)
            if os.path.isfile(output_path) and not file.startswith("Syllabus" + course_name):
                print("File already exists. Skipping...")
                app.logger.info(f"File already exists. Skipping... {file}")
                continue
            else:
                print("File does not exist (or it maybe the Syllabus file.) Processing...")
                app.logger.info(f"File does not exist. Processing... {file}")            
                file_path = os.path.join(folder, file)
                try:
                    df_chunks = pd.read_csv(file_path, encoding='utf-8', escapechar='\\')
                except Exception as e:
                    print(f"Failed to read file: {file_path}. Error: {e}")
                    app.logger.error(f"Failed to read file: {file_path}. Error: {e}")
                    continue

                print(f"Loaded: {file_path}")
                app.logger.info(f"Loaded: {file_path}")

                input_text_list = df_chunks.iloc[:, 1].tolist()

                # Skip embedding if the input text list is empty
                if not input_text_list:
                    print(f"Skipping empty input text for {file_path}")
                    app.logger.info(f"Skipping empty input text for {file_path}")
                    continue

                total_tokens = sum([len(text.split()) for text in input_text_list])
                if total_tokens <= MAX_TOKENS_PER_BATCH:
                    try:
                        embeddings = embed_input_text(input_text_list)
                    except openai.error.InvalidRequestError as e:
                        print(f"Error with input: {input_text_list}")
                        app.logger.error(f"Error with input: {input_text_list}; with Error {e}")
                        continue
                else:
                    # Splitting into multiple batches
                    embeddings = []
                    start = 0
                    tokens_so_far = 0
                    for i, text in enumerate(input_text_list):
                        tokens = len(text.split())
                        if tokens_so_far + tokens > MAX_TOKENS_PER_BATCH:
                            try:
                                embeddings.extend(embed_input_text(input_text_list[start:i]))
                            except openai.error.InvalidRequestError as e:
                                print(f"Error with input batch: {input_text_list[start:i]}")
                                continue
                            time.sleep(WAIT_SECONDS)  # Wait before next batch
                            start = i
                            tokens_so_far = 0
                        tokens_so_far += tokens

                    try:
                        # Embedding the last remaining batch
                        embeddings.extend(embed_input_text(input_text_list[start:]))
                    except openai.error.InvalidRequestError as e:
                        print(f"Error with input batch: {input_text_list[start:]}")
                        app.logger.error(f"Error with input batch: {input_text_list[start:]}; with Error {e}")
                        continue

                # Convert to NumPy array and save
                embeddings_array = np.vstack([np.array(e['embedding']) for e in embeddings])
                
                np.save(output_path, embeddings_array)
                app.logger.info(f"Saved embeddings to {output_path}")
                print(f"Saved embeddings to {output_path}")

    #if __name__ == "__main__":
        # Uncomment the below line if settings.txt is required
        # settings = read_settings("settings.txt")

        # Embed documents for a given course name. Replace 'CourseName' with the actual course name
        #embed_documents_given_course_name('CourseName')

#!/usr/bin/env python3
import sys
import time
from tqdm import tqdm
import openai
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import pickle
from dotenv import load_dotenv

__version__ = '0.0.2'

def save_conversation_history(messages):
    with open('~/conversation_history.pkl', 'wb') as f:
        pickle.dump(messages, f)

def load_conversation_history():
    try:
        with open('~/conversation_history.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []

def call_openai_with_retry(messages):
    retries = 5
    for i in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=messages,
                max_tokens=50,
                temperature=0.4,
            )
            return response
        except openai.error.RateLimitError:
            wait_time = 2 ** i
            print(f"Rate limit hit, retrying after {wait_time} seconds.")
            time.sleep(wait_time)
    raise Exception("Failed after retries")

def get_embedding(text_string):
    response = openai.Embedding.create(
      model="text-embedding-ada-002",
      input=text_string
    )
    return response['data'][0]['embedding']

def chunk_text(text_string, chunk_size=1000):
    return [text_string[i:i+chunk_size] for i in range(0, len(text_string), chunk_size)]

def generate_context(content, chunk_size, user_question):
    messages = []
    chunks = chunk_text(content, chunk_size)
    for chunk in chunks:
        messages.append({
            "role": "system", 
            "content": f"The user will ask: '{user_question}'. The answer might be in the following data: {chunk}"
        })
    return messages

def generate_context_from_file(file_path, chunk_size, user_question):
    with open(file_path, 'r') as file:
        file_content = file.read()
    return generate_context(file_content, chunk_size, user_question)

def generate_context_from_string(string, chunk_size, user_question):
    return generate_context(string, chunk_size, user_question)

def get_all_files(exclude_dirs, extensions, recursive, verbose=True):
    all_files = []
    if verbose:
        print("Starting file listing. This might take a while if there are a lot of directories...")
    with tqdm(desc="Listing files", disable=not verbose) as pbar:
        for dirpath, dirnames, filenames in os.walk(os.getcwd()):
            pbar.update(1)
            if any(dirpath.startswith(edir) for edir in exclude_dirs):
                continue
            for filename in filenames:
                if extensions:
                    if any(filename.endswith(ext) for ext in extensions):
                        filepath = os.path.join(dirpath, filename)
                        all_files.append(filepath)
                else:
                    filepath = os.path.join(dirpath, filename)
                    all_files.append(filepath)
            if not recursive:
                break
    return all_files

def load_or_generate_embeddings(all_files, verbose=True):
    try:
        with open('~/embeddings.pkl', 'rb') as f:
            file_embeddings = pickle.load(f)
    except FileNotFoundError:
        file_embeddings = {}

    total_files = len(all_files)
    with tqdm(total=total_files, desc="Generating embeddings", disable=not verbose) as pbar:
        for filepath in all_files:
            try:
                current_timestamp = os.path.getmtime(filepath)
                if filepath not in file_embeddings or file_embeddings[filepath][2] != current_timestamp:
                    with open(filepath, 'r') as file:
                        file_content = file.read()
                        chunks = chunk_text(file_content)
                        embeddings = generate_embeddings(chunks)
                        for i, embedding in enumerate(embeddings):
                            file_embeddings[filepath] = (i, embedding, current_timestamp)
                            pbar.update(1)
            except:
                pbar.set_postfix_str(f"Skipped file {filepath}.")  # Skip files that can't be read as text

    for filepath in list(file_embeddings):  # Use list to avoid changing the dictionary size during iteration
        if not os.path.exists(filepath):
            del file_embeddings[filepath]

    # Save embeddings to local database
    with open('~/embeddings.pkl', 'wb') as f:
        pickle.dump(file_embeddings, f)

    return file_embeddings

def generate_embeddings(chunks):
    with concurrent.futures.ThreadPoolExecutor(max_workers=60) as executor:
        futures = {executor.submit(get_embedding, chunk) for chunk in chunks}
        embeddings = []
        for future in concurrent.futures.as_completed(futures):
            try:
                embeddings.append(future.result())
            except Exception as exc:
                print(f'An exception occurred: {exc}')
    return embeddings

def generate_context_from_files(file_embeddings, user_question):
    messages = []
    query_embedding = get_embedding(user_question)

    # Calculate the similarity between the query embedding and each file embedding
    similarities = []
    for filepath, (chunk_index, chunk_embedding, current_timestamp) in file_embeddings.items():
        similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
        similarities.append((filepath, chunk_index, similarity))

    # Sort by similarity and select the top 20 most similar file chunks
    similarities.sort(key=lambda x: x[2], reverse=True)
    top_similarities = similarities[:20]

    # Include the contents of the top similar file chunks as context
    parts = []
    for filepath, chunk_index, similarity in top_similarities:
        with open(filepath, 'r') as file:
            file_content = file.read()
            chunks = chunk_text(file_content)
            selected_chunk = chunks[chunk_index].strip()  # Remove leading and trailing whitespace, including new lines
            parts.append(selected_chunk)
    context = ', '.join(f'"{part}"' for part in parts)
    messages.append({"role": "system", "content": f"The user will ask: '{user_question}'. The answer might be in the following data: {context}"})
    return messages

def main():
    # Load environment variables from .env.
    load_dotenv()

    # Set your OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate embeddings for files and find the most similar ones to a query.')
    parser.add_argument('question', help='The user question.')
    parser.add_argument('--show-history', action='store_true', help='Show conversation history.')
    parser.add_argument('--no-context', action='store_true', help='Ask the question without any context.')
    parser.add_argument('--recursive', action='store_true', help='Enable recursive search. If not provided, the search will be limited to the current directory.')
    parser.add_argument('--extensions', nargs='*', default=[], help='A list of file extensions to include.')
    parser.add_argument('--exclude', nargs='*', default=[], help='A list of directories to exclude.')
    parser.add_argument('--file', default=None, help='Path to a text file to use as context.')
    parser.add_argument('--string', default=None, help='A string to use as context.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--version', action='version', version=f'Sol v{__version__}')
    args = parser.parse_args()

    # Get the user's question from the command line arguments
    user_question = args.question

    # Load conversation history
    messages = load_conversation_history()

    # Show conversation history if --show-history flag is set
    if args.show_history:
        user_counter = 1
        assistant_counter = 1
        # Take the 10 most recent messages
        recent_messages = messages[-10:]
        for message in recent_messages:
            role = message['role']
            content = message['content']
            if role == 'system':
                continue
            elif role == 'user':
                print(f"User Message {user_counter}:")
                user_counter += 1
            elif role == 'assistant':
                print(f"Assistant Message {assistant_counter}:")
                assistant_counter += 1
            print(f"  {content}\n")

    # If there's no conversation history, start a new conversation
    if len(messages) == 0:
        messages.append({"role": "system", "content": "You are a helpful CLI assistant, so advanced that you typically know the answer before the user asks the question."})

    # If a file path is provided, generate context from file
    if args.file is not None:
        file_messages = generate_context_from_file(args.file, user_question)
        messages.extend(file_messages)

    # If a string is provided, generate context from string
    elif args.string is not None:
        string_messages = generate_context_from_string(args.string, user_question)
        messages.extend(string_messages)

    # If neither file nor string is provided, generate context from files in the directory tree
    else:
        verbose = not os.path.exists('~/embeddings.pkl')

        all_files = get_all_files(args.exclude, args.extensions, args.recursive, args.verbose)
        file_embeddings = load_or_generate_embeddings(all_files, args.verbose)
        file_messages = generate_context_from_files(file_embeddings, user_question)
        messages.extend(file_messages)

    # Add the user's question to the messages
    messages.append({"role": "user", "content": user_question})
    #print(messages)
    # Generate a completion using OpenAI's chat-based language model
    try:
        response = call_openai_with_retry(messages)

        # Retrieve and print the assistant's reply
        assistant_reply = response.choices[0].message['content']
        print()
        print(assistant_reply)
        
        # Save conversation history
        messages.append({"role": "assistant", "content": assistant_reply})
        save_conversation_history(messages)

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()

import re
import pandas as pd
import evadb
import os
import tokenize
import io
import numpy as np
import warnings
from evadb.configuration.constants import EvaDB_INSTALLATION_DIR
import uuid
from git import Repo
import tempfile
import shutil
from datetime import datetime
import requests
import sys

from openai import OpenAI
os.environ['OPENAI_API_KEY'] = ""
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'], 
)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
 
cursor = evadb.connect().cursor()

def get_tokens(code):
    """
    Returns a string of tokens from the given code snippet.
    :param code: String, the code snippet.
    :return: String, the tokens.
    """

    tokens = []
    for tok in tokenize.tokenize(io.BytesIO(code.encode('utf-8')).readline):
        tokens.append(tok.string)
    return ' '.join(tokens)

def preprocess_python_code(code):
    """
    Preprocesses the given Python code snippet.
    :param code: String, the code snippet.
    :return: String, the preprocessed code snippet.
    """

    code = re.sub(r'#.*?\n', '\n', code)
    code = re.sub(r"'''(.*?)'''", '', code, flags=re.DOTALL)  
    code = re.sub(r'"""(.*?)"""', '', code, flags=re.DOTALL) 

    # Remove single quotes 
    code = code.replace("'", "")
    return code.strip()

# Generate embeddings with OpenAI
def get_code_embedding(code):
    """
    Generates an embedding for the given code snippet.
    :param code: String, the code snippet.
    :return: NDArray, the embedding.
    """
    response = client.embeddings.create(input=code, model="text-embedding-ada-002")
    embedding = response.data[0].embedding
    embedding = np.array(embedding).reshape(1,-1)
    return embedding


def insert_code_embedding(file_name, code_snippet):
    """
    Inserts a code embedding into the database.
    :param file_name: String, the file name.
    :param code_snippet: String, the code snippet.
    :return: None.
    """
    embedding = get_code_embedding(code_snippet).tolist()
    cursor.query(f"""
        INSERT INTO code_embeddings_table (file_name, code_snippet, embedding)
        VALUES ('{file_name}','{code_snippet}', '{embedding}');
    """).df()


def process_files_in_cloned_repo(repo_path):
    """
    Processes all the Python files in the cloned repository.
    :param repo_path: String, the path to the cloned repository.
    :return: List, the processed files.
    """
    processed_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):  # Check if it's a Python file
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as file_content:
                    content = file_content.read()

                    # Process and create embeddings for the file content
                    code_snippet = preprocess_python_code(content)
                    print(f"INSERTING CODE EMBEDDING FOR: {file_path}")
                    insert_code_embedding(file, code_snippet)

                    processed_files.append({
                        "path": file_path,
                        "content": content
                    })

    return processed_files
def post_pull_request_comment(owner, repo, pr_number, comment, token):
    """
    Posts a comment on a specific pull request.

    :param owner: String, the GitHub username or organization name that owns the repository.
    :param repo: String, the repository name.
    :param pr_number: Integer, the pull request number.
    :param comment: String, the comment text to post.
    :param token: String, the GitHub API token.
    :return: The response from the GitHub API (JSON).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"body": comment}
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def commit_changes(repo_path,file_path, commit_message):
    """ 
    Commits the changes to the cloned repository.
    :param repo_path: String, the path to the cloned repository.
    :param file_path: String, the path to the file to commit.
    :param commit_message: String, the commit message.
    :return: None."""
    repo = Repo(repo_path)
    repo.git.add(file_path) 
    repo.index.commit(commit_message)

def push_changes_to_new_branch(repo):
    """
    Pushes the changes to a new branch.
    :param repo: Repo, the cloned repository.
    :return: String, the name of the new branch.
    """
    branch_name = f"update-embeddings-{uuid.uuid4()}"
    origin = repo.remote(name='origin')
    repo.git.checkout('HEAD', b=branch_name) 
    origin.push(branch_name)
    return branch_name



def create_pull_request(token, owner, repo, branch_name, title, body):
    """
    Creates a pull request.
    :param token: String, the GitHub API token.
    :param owner: String, the GitHub username or organization name that owns the repository.
    :param repo: String, the repository name.
    :param branch_name: String, the name of the branch to create the pull request from.
    :param title: String, the title of the pull request.
    :param body: String, the body of the pull request.
    :return: The response from the GitHub API (JSON)."""
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    data = {
        'title': title,
        'head': branch_name,
        'base': 'main',  # or 'master' or your default branch
        'body': body
    }
    response = requests.post(
        f'https://api.github.com/repos/{owner}/{repo}/pulls',
        headers=headers,
        json=data
    )
    return response.json()


def create_tables(cursor):
    """
    Creates the tables and functions in the database.
    
    tables:
    - code_embeddings_table: Stores the code embeddings.
    - response: Stores the response from the GPT-3 model.
    - analyze: Stores the analysis from the GPT-3 model.

    functions:
    - STR2ARRAY: Converts a string to an array.
    - EmbeddingArrayConverter: Converts an embedding array to a string.
    :return: None.
    """

    cursor.query("DROP FUNCTION IF EXISTS STR2ARRAY;").df()
    cursor.query(f"""
        CREATE FUNCTION STR2ARRAY
        IMPL '{EvaDB_INSTALLATION_DIR}/functions/STR2ARRAY.py'
    """).df()

    cursor.query("DROP FUNCTION IF EXISTS EmbeddingArrayConverter;").df()
    cursor.query(f"""
        CREATE FUNCTION EmbeddingArrayConverter
        IMPL '{EvaDB_INSTALLATION_DIR}/functions/EmbeddingArrayConverter.py'
    """).df()
    cursor.query("""
        DROP TABLE IF EXISTS code_embeddings_table
    """).df()

    cursor.query("""
        CREATE TABLE code_embeddings_table (file_name TEXT(100), code_snippet TEXT(1000), embedding NDARRAY FLOAT32(1,1536))
    """).df()


    cursor.query("""
    DROP TABLE IF EXISTS response""").df()

    cursor.query("""
        CREATE TABLE response (
        response TEXT(400))
                    """).df()


    cursor.query("""
    DROP TABLE IF EXISTS analyze""").df()

    cursor.query("""
        CREATE TABLE analyze (
        analysis TEXT(500))
                    """).df()
    print("Tables created successfully")


def main():

    create_tables(cursor)

    current_directory = os.getcwd()

    with tempfile.TemporaryDirectory(dir=current_directory) as tmpdirname:
        print(f"Temporary directory created at {tmpdirname}")
    os.makedirs(tmpdirname, exist_ok=True)
    owner = input("input repo owner here:")
    repo = input("input repo name here:")
    token = input("input github token here:")
    LOCAL_REPO_PATH = tmpdirname 
    BASE_URL = "https://api.github.com"
    REPO_URL = f"https://github.com/{owner}/{repo}.git"


    Repo.clone_from(REPO_URL, LOCAL_REPO_PATH)


    print("Processing files in cloned repo...")
    all_processed_files = process_files_in_cloned_repo(LOCAL_REPO_PATH)


    print("finished processing files")

    # for index, row in query_result.iterrows():
    #   print(f"Code Snippet {index + 1}:\n{row['code_snippet']}\n")



    #After files are created, create the vector index
    cursor.query("""
        CREATE INDEX IF NOT EXISTS code_embedding_index
        ON code_embeddings_table (STR2ARRAY(embedding))
        USING QDRANT;
    """).df()

    user_input_code = input("Describe the bug you're encountering: ")
    user_input = preprocess_python_code(user_input_code)
    user_embedding = get_code_embedding(user_input)


    print("Finding relevant code snippet...")
    code_snippets = cursor.query( f"""
        SELECT code_snippet, file_name FROM code_embeddings_table ORDER BY
        Similarity(
        EmbeddingArrayConverter('{user_embedding}'),
        EmbeddingArrayConverter(embedding)
        ) DESC
        LIMIT 1
    """).df()
    print("Relevant code snippet found.")

    file_name_to_update = code_snippets.iloc[0]['file_name']
    # Execute the query
    code_snippets = code_snippets.iloc[0]['code_snippet']

    cursor.query(f"""INSERT INTO response(response) VALUES ('{code_snippets}');""").df()

    print("Running bug fix/feature check...")
    query = cursor.query("""
        SELECT ChatGPT("Given the following bug in code: '{user_input_code}'. Please provide a corrected version of the entire code snippet. Output should be code only, with no additional explanations or comments.", response) 
        FROM response;
    """).df()

    response = query.at[0, 'response']
    cursor.query(f"""INSERT INTO analyze(analysis) VALUES ('{response}');""").df()

    print("running quality check")
    chatgpt_query = """
        SELECT ChatGPT("Please analyze the following code and provide recommendations for improvement:", analysis)
        FROM analyze;
    """
    chatgpt_result = cursor.query(chatgpt_query).df()

    quality_check_result = chatgpt_result.at[0, 'response']



    #write response to a file

    file_path_to_update = os.path.join(LOCAL_REPO_PATH, file_name_to_update)

    if response.startswith(("import", "def", "class", "#")):
        # Looks like code, write to file
        with open(file_path_to_update, 'w') as file:
            file.write(response)
    else:
        # Output is not code, print the response for the user
        print("Response from GPT:", response)
        shutil.rmtree(tmpdirname)
        sys.exit()

    commit_message = "Update output with bug fix response"
    commit_changes(LOCAL_REPO_PATH,file_path_to_update, commit_message)
    branch_name = push_changes_to_new_branch(Repo(LOCAL_REPO_PATH))

    title = "Automated Bug Fix"
    body = "This pull request contains an automated bug fix."

    pr_response = create_pull_request(token, owner, repo, branch_name, title, body)

    if pr_response and 'html_url' in pr_response:
        print(f"Pull request created successfully. URL: {pr_response['html_url']}")

        # Posting the quality check results as a comment
        pr_number = pr_response.get('number')  # Extract PR number from the response
        comment = f"Code Quality Check Results:\n\n{quality_check_result}"
        comment_response = post_pull_request_comment(owner, repo, pr_number, comment, token)
        if comment_response and 'html_url' in comment_response:
            print(f"Comment posted successfully. URL: {comment_response['html_url']}")
        else:
            print("Failed to post comment. Response:", comment_response)
    else:
        print("Failed to create pull request. Response:", pr_response)


    shutil.rmtree(tmpdirname)
    
if __name__ == "__main__":
    main()

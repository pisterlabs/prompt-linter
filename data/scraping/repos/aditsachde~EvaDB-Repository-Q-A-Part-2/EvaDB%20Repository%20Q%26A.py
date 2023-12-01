# Necessary imports
import csv
import evadb
import mimetypes
import nbformat
from openai import OpenAI
import os
import shutil
import subprocess
import tempfile
import tiktoken

# Connect to an EvaDB instance
cursor = evadb.connect().cursor()

# Set env vars for both the OpenAI library and EvaDB, plus, create an OpenAI API client
os.environ['OPENAI_KEY'] = 'sk-replace-me'
os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_KEY']
client = OpenAI()

# Custom logic to load a repository.
# Downloads the entire repo, extracts texts from notebooks, and generates embeddings
def load_repository(cursor, repo_url):
  temp_dir = tempfile.mkdtemp()

  target_directory = "repo"
  git_clone_command = ["git", "clone", repo_url, target_directory]
  subprocess.check_call(git_clone_command, cwd=temp_dir)
  repo_path = os.path.join(temp_dir, target_directory)

  id = 1
  rows = [['id', 'name', 'text', 'embeddings']]

  for root, dirs, files in os.walk(repo_path):
      dirs[:] = [d for d in dirs if not d.startswith('.')]
      files = [f for f in files if not f.startswith('.')]

      for file in files:
        if not any(d.startswith('.') for d in root.split(os.path.sep)):
          file_path = os.path.join(root, file)
          mime_type, _ = mimetypes.guess_type(file_path)
          rel_path = os.path.relpath(os.path.join(root, file), repo_path)
          is_text_file = mime_type and mime_type.startswith('text/')

          if is_text_file:
            with open(file_path, 'r', encoding='utf-8') as file:
              file_content = file.read()

          elif file_path.endswith('.ipynb'):
            with open(file_path, 'r', encoding='utf-8') as file:
              notebook_content = nbformat.read(file, as_version=4)
            file_content = ''
            for cell in notebook_content['cells']:
              if cell.cell_type == 'markdown' or cell.cell_type == 'code':
                file_content += cell.source
                file_content += "\n\n"
          else:
            break

          embedding_text = file_content.replace("\n", " ")

          encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
          token_count = len(encoding.encode(embedding_text))

          # Embedding token limit, gets hit by script files
          if token_count < 4097:
            # The text-embedding-ada-002 model is recommended by OpenAI for all usecases.
            embedding_response = client.embeddings.create(input = [embedding_text], model="text-embedding-ada-002")
            embeddings = embedding_response.data[0].embedding
          else:
            embeddings = []

          rows.append([id, rel_path, file_content, embeddings])
          id += 1


  csv_file = os.path.join(temp_dir, "output.csv")

  with open(csv_file, mode="w", newline="") as file:
      writer = csv.writer(file, delimiter=",")
      for row in rows:
          writer.writerow(row)

  cursor.query('''
  DROP TABLE IF EXISTS repository
  ''').df()

  cursor.query('''
  CREATE TABLE repository
  (id INTEGER,
  name TEXT(150),
  text TEXT(150000),
  embeddings TEXT(150000))
  ''').df()

  cursor.query(f'''
  LOAD CSV '{csv_file}' INTO repository
  ''').df()

  shutil.rmtree(temp_dir)

# Load in an example repository
load_repository(cursor, "https://github.com/microsoft/AI-For-Beginners.git")

# Selects all the items from the repository table.
print(cursor.query('''
SELECT * FROM repository
''').df())

# Load the Embeddings custom AI function for EvaDB
print(cursor.query('''
DROP FUNCTION IF EXISTS Embeddings;
''').df())

print(cursor.query('''
CREATE FUNCTION Embeddings
IMPL  'embeddings.py';
''').df())

# Using the loaded function, figure out which files are the most relevant for the current question
print(cursor.query('''
SELECT name, text, Embeddings("What are the Principles of Responsible AI?", embeddings) FROM repository ORDER BY distance DESC LIMIT 5;
''').df())

# Using a nested statement, figure out which file from the repository is best suited to answer this question,
# and provide that as context to ChatGPT to answer the question
print(cursor.query('''
SELECT ChatGPT('What are the Principles of Responsible AI?', s.text) FROM
(
    SELECT Embeddings("What are the Principles of Responsible AI?", embeddings), name, text FROM repository ORDER BY distance DESC LIMIT 1
) AS s;
''').df())

# Load the Llama 2 custom AI function for EvaDB
print(cursor.query('''
DROP FUNCTION IF EXISTS EvaLlama;
''').df())

print(cursor.query('''
CREATE FUNCTION EvaLlama
IMPL  'llama.py';
''').df())

# Using a nested statement, figure out which file from the repository is best suited to answer this question,
# and provide that as context to Llama 2 to answer the question
print(cursor.query('''
SELECT EvaLlama('what are the five Principles of Responsible AI?', s.text) FROM
(
    SELECT Embeddings("what are the five Principles of Responsible AI?", embeddings), name, text FROM repository ORDER BY distance DESC LIMIT 1
) AS s;
''').df())
import os
import frontmatter
import sqlite3
import openai
import json

# define path and API key
path = "~/Documents/websites/swizec.com/src/pages/blog"
openai.api_key = "Your OpenAI Key"

# connect to SQLite database
conn = sqlite3.connect('embedding_vectors.db')
cursor = conn.cursor()

# create table if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS vectors (
        path TEXT PRIMARY KEY,
        filename TEXT,
        title TEXT,
        vector JSON
    )
''')

# walk through the directories
for root, dirs, files in os.walk(os.path.expanduser(path)):
    for file in files:
        if file == "index.mdx":
            # get the full file path
            full_path = os.path.join(root, file)
            print(f'Processing {full_path}')

            # read the file
            with open(full_path, 'r') as f:
                try:
                    post = frontmatter.load(f)
                    title = post.get('title', 'No Title')
                except Exception as e:
                    print(f'Error parsing file {full_path}: {e}')
                    continue

                # get the embedding
                try:
                    response = openai.Embedding.create(
                        model="text-embedding-ada-002",
                        input=f.read()
                    )
                    embedding = response['data'][0]['embedding']
                except Exception as e:
                    print(f'Error generating embedding for {full_path}: {e}')
                    continue

                # save the embedding
                cursor.execute('''
                    INSERT INTO vectors (path, filename, title, vector) 
                    VALUES (?, ?, ?, ?)
                ''', (full_path, file, title, json.dumps(embedding)))

# commit changes and close connection
conn.commit()
conn.close()

print('Done.')

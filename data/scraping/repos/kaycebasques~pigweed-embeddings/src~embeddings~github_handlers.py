import os

import openai

import database
import utilities

def embed(mgr, url, text, checksums):
    openai_client = openai.OpenAI(api_key=os.environ.get('OPENAI_KEY'))
    db = database.Database()
    if utilities.token_count(text) > utilities.max_token_count():
        return
    if db.row_exists(content=text):
        db.update_timestamp(content=text)
    else:
        embedding = openai_client.embeddings.create(
            input=text,
            model=utilities.embedding_model()
        ).data[0].embedding
        db.add(content=text, content_type='code', url=url, embedding=embedding)

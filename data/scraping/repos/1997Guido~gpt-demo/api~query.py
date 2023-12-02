# services.py
import openai
from .models import QueryCategory

class DatabaseQueryGenerator:
    def __init__(self, db_description, api_key):
        self.db_description = db_description
        openai.api_key = api_key

    def get_sql_query(self, question):
        prompt = f"{self.db_description}\nQuestion: {question}"
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=prompt,
        )
        sql_query = response['choices'][0]['text'].strip()
        return sql_query
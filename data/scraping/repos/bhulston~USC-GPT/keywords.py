import os
from dotenv import find_dotenv, load_dotenv
import openai

load_dotenv(find_dotenv())
openai.api_key = os.environ.get("OPENAI_API")

def keyword_agent(query):

    system_prompt = """
    You are an expert at creating keywords/phrases for vector searches in databases. You take human queries and optimize them to find the best match in a vector database based on similarity search.
    You will be given a query from a user asking for help in searching a vector database meeting their needs. 
    Keep in mind that that the vector database contains several documents, where each document is a class that the user could take.

    Your goal is to output (only) a string (with no other output aside from it) consisting of keywords or phrases that will optimize the query. You will build this string, keeping in mind that the search will be run against college descriptions of class information.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    return response["choices"][0]["message"]["content"]


import tiktoken
import os
from dotenv import load_dotenv
import openai

load_dotenv("../.env")

openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")

# encoding = tiktoken.get_encoding("cl100k_base")
# encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# print(len(encoding.encode("Hello world, this is fun!Hello world, this is fun!Hello world, this is fun!Hello world, this is fun!Hello world, this is fun!Hello world, this is fun!Hello world, this is fun!Hello world, this is fun!Hello world, this is fun!Hello world, this is fun!")))

# Open the file with information about movies
movie_data = os.path.join(os.getcwd(), "movies.csv")
content = open(movie_data, "r").read()

# Use tiktoken to tokenize the content and get a count of tokens used.
# encoding = tiktoken.get_encoding("p50k_base")
# print (f"Token count: {len(encoding.encode(content))}")

query = "What's the highest rated movie from the following list\n"
query += "CSV list of movies:\n"
query += content

print (f"{query[0:500]} ...[{len(query)} characters]")

r = openai.ChatCompletion.create(
    model = os.getenv("OPENAI_COMPLETION_MODEL"),
    deployment_id = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
    messages = [{"role" : "assistant", "content" : query}],
)

print (r)
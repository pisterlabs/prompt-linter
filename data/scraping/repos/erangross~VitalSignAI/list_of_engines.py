import openai
import os
openai.api_key = os.environ.get("OPENAI_API_KEY")

engines = openai.Engine.list()

for engine in engines["data"]:
    print(engine["id"])

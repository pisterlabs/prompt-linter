# %%
import json
import os
import pickle

import jsonlines
import openai
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")


with open("./data/papers_full.pickle", "rb") as handle:
    data = pickle.load(handle)

docs = "\n".join(
    [
        doc["title"].replace("\n", " ")
        + " "
        + doc["abstract"].replace("\n", " ")
        for doc in data
    ]
)

request = {"text": f"{docs}"}

print(request)

file_path = "./data/file.txt"

with jsonlines.open(file_path, mode="w") as writer:
    # writer.write(json.dumps({"text": docs}))
    writer.write({"text": docs})


# openai.File.create(file=open(file_path), purpose="search")
# %%
# doesn't work : <
openai.Engine("davinci").search(
    file="fileID",
    query="Which article is from the Reinforcement Learning field?",
)

# %%
openai.File.list()
# openai.File("fileID").delete()

import os
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from timer import Timer
from langchain.chains import LLMChain
from transformers import pipeline
import pandas as pd
import json
import zipfile
import json
import zipfile
import pandas as pd

REQUESTED = os.environ.get("REQUESTED")
if not REQUESTED:
    raise ValueError("REQUESTED environment variable is not set or is empty.")

json_data = None

with zipfile.ZipFile("/tmp/iexec_in/protectedData.zip") as myzip:
    for filename in myzip.namelist():
        if filename.endswith('.json'):
            with myzip.open(filename) as file:
                json_data = file.read().decode("utf-8")

parsed_data = json.loads(json_data.replace("'", "\""))
keys = list(parsed_data['data'][0].keys())
data = {}

for key in keys:
    values = [entry[key] for entry in parsed_data['data']]
    data[key] = values

table = pd.DataFrame.from_dict(data)
print(table)

tqa = pipeline(task="table-question-answering",
               model="google/tapas-large-finetuned-wtq")


def ask_question(question):
    result = tqa(table=table, query=question)
    result_json = json.dumps(result)

    with zipfile.ZipFile("/tmp/iexec_out/result.zip", "w") as myzip:
        myzip.writestr("result.json", result_json)

    print("Result saved to /tmp/iexec_out/result.zip")


with Timer():
    ask_question(REQUESTED)

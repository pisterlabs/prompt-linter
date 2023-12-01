from fastapi import FastAPI
from pydantic import BaseModel
import json
import os,time
import subprocess
import tempfile
#from langchain.llms import OpenAI
from typing import Any, Dict

class Input(BaseModel):
    prompt: str
    featureName: str


def run_cmd(prompt_text):
    start_time = time.time()

    # Create a temporary file and write the prompt_text into it
    prompt_fd, prompt_path = tempfile.mkstemp()
    pyfile_path="tmp"
    try:
        with os.fdopen(prompt_fd, 'w') as prompt_file:
            print (f"{prompt_text}")
            prompt_file.write(prompt_text)
        with open(pyfile_path, 'w') as prompt_file:
            print ("")
            prompt_file.write("prompt_text")

        p2 = subprocess.Popen(["bito", "-p", prompt_path, "-f", pyfile_path], stdout=subprocess.PIPE)  # added stdout=subprocess.PIPE
        output, _ = p2.communicate()  # Get output once here
    except subprocess.CalledProcessError as e:
        print(f"Subprocess returned error: {e.output}")
        output = e.output
    finally:
        # Ensure the temporary file is deleted even if an error occurs
        os.unlink(prompt_path)

    end_time = time.time()
    total_time = end_time - start_time

    return output  # Return the stored output, decoded from bytes to string



def prompt_to_query_new(prompt: str, info: str, data: str):
    template = """
    Your mission is convert SQL query from given request
 {prompt}
Do not include featureName in the output sql query.
Use following database information for this purpose (info key is a database column name and info value is explanation) : {info} .
along with this i am sharing some sample data from this table :  {data}.
Create aggregation sql query on mapped column if you see aggregated by or categorized by keyword in input.
    --------
    Put your query in the  JSON structure with key name is 'query'
    ONLY SELECT sql query is expected from the response.
    No Other Explanations / Other Text is required.
    """
    final_prompt = template.format(prompt=prompt[], info=info, data=data)
    output = run_cmd(final_prompt)
    return output

def read_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        data = file.read()
    return data

app = FastAPI()

@app.post("/process")
async def process(input: Input):
    info = read_file( f"{input.featureName}_info.json")
    data = read_file( f"{input.featureName}_data.txt")
    query = prompt_to_query_new(input.prompt, info, data)
    #query_json = json.loads(query)
    return {"Generated SQL Query": query}

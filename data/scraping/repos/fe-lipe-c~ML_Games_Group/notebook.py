import requests
import json
import os

# from langchain.document_loaders import MathpixPDFLoader


# get openai api key from environment
openai_api_key = os.environ["OPENAI_API_KEY"]
mathpix_api_key = os.environ["MATHPIX_API_KEY"]
mathpix_api_id = os.environ["MATHPIX_API_ID"]

mathpix_url = "https://api.mathpix.com/v3/pdf"
# pdf_url = "http://cs229.stanford.edu/notes2020spring/cs229-notes4.pdf"

full_path = os.getcwd()
list_files = os.listdir("data/raw")
list_files[2]
file_path = full_path + "/data/raw/" + list_files[2]
file_path

options = {
    "conversion_formats": {"tex.zip": True},
    "math_inline_delimiters": ["$", "$"],
    "rm_spaces": True,
}

result = requests.post(
    mathpix_url,
    headers={
        "app_id": mathpix_api_id,
        "app_key": mathpix_api_key,
    },
    data={
        "options_json": json.dumps(options),
    },
    files={"file": open(file_path, "rb")},
)

dict_string = json.dumps(result.json(), indent=4, sort_keys=True)
pdf_id = json.loads(dict_string)["pdf_id"]
pdf_id

result_url = mathpix_url + "/" + pdf_id + ".tex"
response_result = requests.get(
    result_url,
    headers={
        "app_key": mathpix_api_key,
        "app_id": mathpix_api_id,
    },
)

with open("multiagent_new" + ".tex.zip", "wb") as f:
    f.write(response_result.content)


pdf_file = "1.invariant_causal_prediction_for_block_mdps.pdf"
loader = MathpixPDFLoader(file_path)

loader.load()

headers = {
    "app_id": "felipe_c1fd88",
    "app_key": "MATHPIX_API_KEY",
    "Content-type": "application/json",
}

service = "https://api.mathpix.com/v3/md"

response = requests.post(url, headers=headers)

response.json()

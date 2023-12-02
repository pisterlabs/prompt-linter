import json
from typing import Dict

from langchain.llms.sagemaker_endpoint import LLMContentHandler


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": [[{"role": "user", "content": prompt}]], **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        if len(response_json) == 0:
            raise Exception("Response not found")
        return response_json[0]['generation']['content'].strip()


def format_sagemaker_inference_data(data):
    split_key = '\n\n'
    question = (data_list := data.split(split_key))[0].replace("Question:", '').strip()
    query = data_list[1].replace("SQLQuery:", '').strip()
    query_explanation = split_key.join(data_list[2:]).replace("Explanation:", '').strip()
    return question, query, query_explanation

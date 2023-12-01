from langflow import CustomComponent
from langchain.schema import Document
from typing import Optional
from langchain.llms.base import BaseLLM

from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials

class Watsonx(CustomComponent):
    display_name = "Watsonx"
    description = "Watsonx foundation models"
    beta = False

    def build_config(self) -> dict:
        model_options = [
            # Part of the officially supported foundation models in watsonx.ai
            "bigcode/starcoder",
            "bigscience/mt0-xxl",
            "eleutherai/gpt-neox-20b",
            "google/flan-t5-xxl",
            "google/flan-ul2",
            "ibm/mpt-7b-instruct2",
            "meta-llama/llama-2-70b-chat"
        ]

        decoding_options = [
            "sample",
            "greedy"
        ]

        return {
            "model_name": {
                "display_name": "Model Name",
                "options": model_options,
                "value": model_options[0],
                "info": "The ID of the model or tune to be used for this request."
            },
            "api_endpoint": {
                "display_name": "API Endpoint",
                "info": "api endpoint for watsonx"
            },
            "api_key": {
                "display_name": "API Key",
                "password": True,
                "info": "API uses API keys for authentication."
            },
            # "api_endpoint": { "display_name": "API Endpoint", "value": "https://bam-api.res.ibm.com/v1/" },
            "decoding_method": {
                "display_name": "Decoding Method",
                "options": decoding_options,
                "value": decoding_options[0],
                "info": "Represents the strategy used for picking the tokens during generation of the output text."
            },
            "max_new_tokens": {
                "display_name": "Max New Tokens",
                "value": 1500,
                "info": "The maximum number of new tokens to be generated."
            },
            "temperature": {
                "display_name": "Temperature",
                "value": 1,
                "info": "A value used to modify the next-token probabilities in sampling mode."
            },
            "code": {"show": False}
        }

    def build(
        self,
        model_name: str,
        api_key: str,
        api_endpoint: str,
        decoding_method: str,
        max_new_tokens: int,
        temperature: int,
    ) -> BaseLLM:
        creds = Credentials(api_key, api_endpoint)
        params = GenerateParams(decoding_method=decoding_method, max_new_tokens=max_new_tokens, temperature=temperature)
        model = LangChainInterface(model=model_name, params=params, credentials=creds)
        return model

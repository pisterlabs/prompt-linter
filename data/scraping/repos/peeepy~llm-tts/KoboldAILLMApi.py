import langchain
from langchain.llms.base import LLM, Optional, List, Mapping, Any
import requests
import json
from pydantic import Field
    
with open('../llm-tts/config.json') as file:
    config_data = json.load(file)

API_URL = config_data['API_URL']
s = requests.Session()
persona = config_data['persona']

class KoboldApiLLM(LLM):
    endpoint: str = Field(...)

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]]=None) -> str:
        # Prepare the JSON data
        data = {
            "prompt": prompt,
            "max_context_length": 2048,
            "max_length": 100,
            "rep_pen": 1.15,
            "rep_pen_range": 2048,
            "rep_pen_slope": 0.1,
            "temperature": 1.35,
            "tfs": 0.69,
            "top_a": 0,
            "top_p": 1,
            "top_k": 0,
            "typical": 1,
        }

        # Add the stop sequences to the data if they are provided
        if stop is not None:
            data["stop_sequence"] = stop

        # Send a POST request to the Kobold API with the data
        response = s.post(f"{self.endpoint}/api/v1/generate", json=data)
        response.raise_for_status()

        # Check for the expected keys in the response JSON
        json_response = response.json()
        if "results" in json_response and len(json_response["results"]) > 0 and "text" in json_response["results"][0]:
            # Return the generated text
            text = json_response["results"][0]["text"].strip().replace("'''", "```")

            # Remove the stop sequence from the end of the text, if it's there
            if stop is not None:
                for sequence in stop:
                    if text.endswith(sequence):
                        text = text[: -len(sequence)].rstrip()


            print(text)
            return text
        else:
            raise ValueError("Unexpected response format from Kobold API")



    def __call__(self, prompt: str, stop: Optional[List[str]]=None) -> str:
        return self._call(prompt, stop)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {'endpoint': self.endpoint} #return the kobold_ai_api as an identifying parameter

llm = KoboldApiLLM(endpoint=API_URL)

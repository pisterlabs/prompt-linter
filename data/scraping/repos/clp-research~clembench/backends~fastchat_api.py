from typing import List, Dict, Tuple, Any
from retry import retry

import json
import openai
import backends

logger = backends.get_logger(__name__)

MAX_TOKENS = 100

# This is less than ideal. We can't really know at this time which models this backend
#  will support, because that depends on what the local fastchat happens to be serving
#  at the moment.. So what this can mean at best is that someone somewhere had served
#  these models at least once.. We might still run into a runtime error when generate
#  is called and the model does not happen to be loaded to be served...
# Convention for model names is to prefix them with "fc-" (for fastchat), to avoid
#  ambiguities when the same model is also available via another backend (such as 
#  huggingface_local). This prefix gets removed before the actual call to the API.
SUPPORTED_MODELS = ["fc-vicuna-13b-v1.5", "fc-vicuna-33b-v1.3", "fc-vicuna-7b-v1.5"]

NAME = "fastchat_openai"


class FastChatOpenAI(backends.Backend):

    def __init__(self):
        creds = backends.load_credentials(NAME)
        #openai.api_base = creds[NAME]["fastchat_ip"]
        self.temperature: float = -1.

    def set_fastchat_url(self):
        self.orig_openai_api_base = openai.api_base
        openai.api_based = self.creds[NAME]["fastchat_ip"]

    def unset_fastchat_url(self):
        openai.api_base = self.orig_openai_api_base

    def list_models(self):
        models = openai.Model.list()
        names = [item["id"] for item in models["data"]]
        names = sorted(names)
        [print(n) for n in names]

    @retry(tries=3, delay=0, logger=logger)
    def generate_response(self, messages: List[Dict], model: str) -> Tuple[str, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: chat-gpt for chat-completion, otherwise text completion
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"

        if model.startswith('fc-'):
            model = model[3:] 

        prompt = messages
        api_response = openai.ChatCompletion.create(model=model, messages=prompt,
                                                    temperature=self.temperature, max_tokens=MAX_TOKENS)
        message = api_response["choices"][0]["message"]
        if message["role"] != "assistant":  # safety check
            raise AttributeError("Response message role is " + message["role"] + " but should be 'assistant'")
        response_text = message["content"].strip()
        response = json.loads(api_response.__str__())

        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS

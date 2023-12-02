from typing import List, Dict, Tuple, Any
from retry import retry

import aleph_alpha_client
import anthropic
import backends

logger = backends.get_logger(__name__)

LUMINOUS_SUPREME_CONTROL = "luminous-supreme-control"
LUMINOUS_SUPREME = "luminous-supreme"
LUMINOUS_EXTENDED = "luminous-extended"
LUMINOUS_BASE = "luminous-base"
SUPPORTED_MODELS = [LUMINOUS_SUPREME_CONTROL, LUMINOUS_SUPREME, LUMINOUS_EXTENDED, LUMINOUS_BASE]

NAME = "alephalpha"


class AlephAlpha(backends.Backend):

    def __init__(self):
        creds = backends.load_credentials(NAME)
        self.client = aleph_alpha_client.Client(creds[NAME]["api_key"])
        self.temperature: float = -1.

    @retry(tries=3, delay=0, logger=logger)
    def generate_response(self, messages: List[Dict], model: str) -> Tuple[Any, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: model name
        :return: the continuation
        """
        assert 0.0 <= self.temperature <= 1.0, "Temperature must be in [0.,1.]"
        prompt_text = ''

        if 'control' in model:
            for message in messages:
                content = message["content"]
                if message['role'] == 'assistant':
                    prompt_text += '### Response:' + content
                elif message['role'] == 'user':
                    prompt_text += '### Instruction:' + content
        else:

            for message in messages:
                if message['role'] == 'assistant':
                    prompt_text += f'{anthropic.AI_PROMPT} {message["content"]}'
                elif message['role'] == 'user':
                    prompt_text += f'{anthropic.HUMAN_PROMPT} {message["content"]}'

            prompt_text += anthropic.AI_PROMPT

        params = {
            "prompt": aleph_alpha_client.Prompt.from_text(prompt_text),
            "maximum_tokens": 100,
            "stop_sequences": ['\n'],
            "temperature": self.temperature
        }

        request = aleph_alpha_client.CompletionRequest(**params)
        api_response = self.client.complete(request=request, model=model)
        response = api_response.to_json()
        response_text = api_response.completions[0].completion.strip()

        prompt = params
        prompt['prompt'] = prompt_text

        return prompt, response, response_text

    def supports(self, model_name: str):
        return model_name in SUPPORTED_MODELS

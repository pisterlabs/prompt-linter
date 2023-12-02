import json
from pdx.logger import logger
from pdx.models.model import Model
from pdx.prompt.prompt_session import PromptSession
from pdx.models.anthropic.client import AnthropicClient
import pdx.models.anthropic.constants as constants
from pdx.models.anthropic.exceptions import handle_anthropic_prompt_validation
from pdx.models.metadata import ModelResponse, ResponseMetadata, ModelTokenUsage


class CompletionModel(Model):
    def __init__(self,
                 api_key: str,
                 model: str,
                 max_tokens: int = 1200,
                 stop: list = ["#", "---"],
                 temperature: float = 0,
                 **kwargs,
                 ):

        self._api_url = "v1/complete"

        self._provider = "anthropic"
        self._client = AnthropicClient(api_key)
        self._model = model
        self._max_tokens_to_sample = max_tokens
        self._stop_sequences = stop
        self._temperature = temperature
        self._top_p = kwargs.get('top_p', -1)
        self._top_k = kwargs.get('top_k', -1)
        self._retries = kwargs.get('retries', 2)

    def _preprocess(self, prompt: PromptSession):
        if prompt.session_type == "chat":
            _prompt = prompt.text_prompt({
                'user': constants.HUMAN_PROMPT,
                'assistant': constants.AI_PROMPT,
                'system': constants.HUMAN_PROMPT,
            })
        else:
            _content = prompt.text_prompt({})
            _prompt = f"{constants.HUMAN_PROMPT} {_content}{constants.AI_PROMPT}"

        handle_anthropic_prompt_validation(_prompt)
        request_params = {
            'prompt': _prompt,
            'stop_sequences': self._stop_sequences,
            'model': self._model,
            'max_tokens_to_sample': self._max_tokens_to_sample,
            'temperature': self._temperature,
            'top_p': self._top_p,
            'top_k': self._top_k,
        }

        return request_params

    def _postprocess(self, response: dict, request_params: dict, request_time) -> ModelResponse:
        _prompt = request_params.pop('prompt', None)

        _r = json.loads(response)
        token_usage = ModelTokenUsage(
            response=None,
            prompt=None,
            total=None)
        response_metadata = ResponseMetadata(
            model=_r['model'],
            api_log_id=_r['log_id'],
            stop=_r['stop'],
            stop_reason=_r['stop'],
            token_usage=token_usage,
            latency=request_time)
        model_response = ModelResponse(
            metadata=response_metadata,
            request_params=request_params,
            data=_r['completion'])

        return model_response

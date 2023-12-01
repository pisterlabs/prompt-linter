import json
from pdx.logger import logger
from pdx.models.model import Model
from pdx.prompt.prompt_session import PromptSession
from pdx.models.cohere.client import CohereClient
from pdx.models.metadata import ModelResponse, ResponseMetadata, ModelTokenUsage


class GenerationModel(Model):
    def __init__(self,
                 api_key: str,
                 model: str,
                 max_tokens: int = 1200,
                 stop: list = [],
                 temperature: float = 0,
                 **kwargs,
                 ):

        self._api_url = "v1/generate"

        self._provider = "cohere"
        self._client = CohereClient(api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        if kwargs.get('stop_sequences', None):
            self._stop_sequences = kwargs.get('stop_sequences', None)
        else:
            self._stop_sequences = stop
        self._end_sequences = kwargs.get('end_sequences', [])
        self._num_generations = kwargs.get('num_generations', 1)
        self._preset = kwargs.get('preset', None)
        self._k = kwargs.get('k', 0)
        self._p = kwargs.get('p', 0.75)
        self._frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        self._presence_penalty = kwargs.get('presence_penalty', 0.0)
        self._return_likelihoods = kwargs.get('return_likelihoods', 'NONE')
        self._logit_bias = kwargs.get('logit_bias', {})
        self._truncate = kwargs.get('truncate', 'END')
        self._retries = kwargs.get('retries', 2)

    def _preprocess(self, prompt: PromptSession):

        _prompt = prompt.text_prompt({})

        request_params = {
            'prompt': _prompt,
            'model': self._model,
            'num_generations': self._num_generations,
            'max_tokens': self._max_tokens,
            'preset': self._preset,
            'temperature': self._temperature,
            'k': self._k,
            'p': self._p,
            'frequency_penalty': self._frequency_penalty,
            'presence_penalty': self._presence_penalty,
            'return_likelihoods': self._return_likelihoods,
            'truncate': self._truncate,
        }

        if self._logit_bias != {}:
            request_params['logit_bias'] = self._logit_bias

        if self._end_sequences != []:
            request_params['end_sequences'] = self._end_sequences

        if self._stop_sequences != []:
            request_params['stop_sequences'] = self._stop_sequences

        return request_params

    def _postprocess(self, response: dict, request_params: dict, request_time) -> ModelResponse:
        _prompt = request_params.pop('prompt', None)
        _r = json.loads(response)

        token_usage = ModelTokenUsage(
            response=None,
            prompt=None,
            total=None)
        response_metadata = ResponseMetadata(
            model=request_params['model'],
            api_log_id=_r['id'],
            warnings=_r['meta'].get('warnings', None),
            token_usage=token_usage,
            latency=request_time)
        model_response = ModelResponse(
            metadata=response_metadata,
            request_params=request_params,
            data=_r['generations'][0]['text'])

        return model_response

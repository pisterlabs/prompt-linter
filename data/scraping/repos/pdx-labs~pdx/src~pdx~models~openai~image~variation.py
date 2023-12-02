import json
from pdx.logger import logger
from pdx.models.model import Model
from pdx.prompt.prompt_session import PromptSession
from pdx.models.openai.client import OpenAIClient
from pdx.models.metadata import ModelResponse, ResponseMetadata
from pdx.models.utils.image import format_response


class ImageVariationModel(Model):
    def __init__(self,
                 api_key: str,
                 **kwargs,
                 ):

        self._api_url = "v1/images/variations"

        self._provider = "openai"
        self._client = OpenAIClient(api_key)
        self._n = kwargs.get('n', 1)
        # Must be one of 256x256, 512x512, or 1024x1024
        self._size = kwargs.get('size', '1024x1024')
        # Must be one of url or b64_json.
        self._response_format = kwargs.get('response_format', 'b64_json')
        self._decode_response = kwargs.get('decode_response', True)
        self._retries = kwargs.get('retries', 2)

    def _preprocess(self, prompt: PromptSession) -> dict:
        request_params = {
            "n": self._n,
            "size": self._size,
            "response_format": self._response_format,
        }
        _files: list = prompt.image_prompt()
        if len(_files) > 1:
            logger.echo('Only one image prompt supported at the moment.')
        request_params['files'] = {
            "image": (_files[0][0], _files[0][1], "image/png")}

        return request_params

    def _postprocess(self, response: dict, request_params: dict, request_time: float) -> ModelResponse:
        _prompt = request_params.pop('prompt', None)
        _files = request_params.pop('files', None)
        _r = json.loads(response)

        response_metadata = ResponseMetadata(
            model='dall-e-variations',
            api_log_id=f"{_r['created']}",
            stop='generation_completed',
            stop_reason='generation_completed',
            latency=request_time)

        _response_data = ['data']
        if len(_response_data) == 1:
            _data = format_response(
                _r['data'][0][request_params['response_format']],
                request_params['response_format'],
                self._decode_response
            )
        elif len(_response_data) > 1:
            _data = [format_response(
                _d[request_params['response_format']],
                request_params['response_format'],
                self._decode_response
            )
                for _d in _r['data']]

        if request_params['response_format'] == 'b64_json':
            _data_type_prefix = 'bytes'
        elif request_params['response_format'] == 'url':
            _data_type_prefix = 'url'
        else:
            _data_type_prefix = 'string'

        _data_type_suffix = ''
        if isinstance(_data, list):
            _data_type_suffix = '_list'

        _data_type = f'image_{_data_type_prefix}{_data_type_suffix}'

        model_response = ModelResponse(
            metadata=response_metadata,
            request_params=request_params,
            data=_data,
            data_type=_data_type
        )

        return model_response

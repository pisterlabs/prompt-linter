from typing import List

import openai
from .embedding import Embedding
from ..contracts import StoreCoreConfig, OpenAIApiType
from ..utils.retry_utils import retry_and_handle_exceptions


def extract_delay_from_rate_limit_error_msg(text):
    import re
    pattern = r'retry after (\d+)'
    match = re.search(pattern, text)
    if match:
        retry_time_from_message = match.group(1)
        return float(retry_time_from_message)
    else:
        return None


class AOAIEmbedding(Embedding):

    @retry_and_handle_exceptions(exception_to_check=openai.error.RateLimitError,
                                 max_retries=5,
                                 extract_delay_from_error_message=extract_delay_from_rate_limit_error_msg)
    def embed(self, text: str) -> List[float]:
        return openai.Embedding.create(
            input=text,
            engine=self.__config.model_name)["data"][0]["embedding"]

    def __init__(self, config: StoreCoreConfig):
        self.__config = config

        openai.api_type = OpenAIApiType.AZURE.value
        openai.api_base = config.model_api_base
        openai.api_version = config.model_api_version

        if config.model_api_key:
            openai.api_key = config.model_api_key.get_value()

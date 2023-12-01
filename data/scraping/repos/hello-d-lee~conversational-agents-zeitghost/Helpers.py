from google.cloud import secretmanager
from decouple import config
import time
import os
from enum import Enum
from google.protobuf import struct_pb2
from langchain import PromptTemplate

_SECRET_ID = 'projects/939655404703/secrets/genai-key'
_SECRET_VERSION = '{}/versions/1'.format(_SECRET_ID)
project = os.environ.get('PROJECT_ID')


def _get_api_key() -> str:
    """
    Retrieve API key from Secret Manager

    :return: str
    """
    sm_client = secretmanager.SecretManagerServiceClient()
    name = sm_client.secret_path(project, _SECRET_ID)
    response = sm_client.access_secret_version(request={"name": _SECRET_VERSION})

    return response.payload.data.decode("UTF-8")


def rate_limit(max_per_minute=15) -> None:
    """
    Rate Limiter to manage API limitations

    :param max_per_minute:

    :return: yield
    """
    period = 60 / max_per_minute
    # print('Waiting')
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            # print(f'Sleeping {sleep_time:.1f} seconds')
            print('.', end='')
            time.sleep(sleep_time)


class VertexModels(Enum):
    """
    Enumeration of available LLM models
    """
    MODEL_EMBEDDING_GECKO = "textembedding-gecko@001"
    MODEL_TEXT_BISON_001 = "text-bison@001"
    MODEL_CHAT_BISON_001 = "chat-bison-001" #TODO UPDATE WITH SDK
    MODEL_CODE_BISON_001 = 'code-bison-001' #TODO UPDATE WITH SDK
    MODEL_TEXT_BISON_ALPHA = "text-bison-alpha" #TODO UPDATE WITH SDK
    

# prompt: python stop a string at first occurance of word in list of words

# Sure, here is the Python code to stop a string at the first occurrence of a word in a list of words:

# ```python
def stop_string_at_first_occurrence(string, words):
    """
    Stops a string at the first occurrence of a word in a list of words.

    Args:
    string: The string to stop.
    words: A list of words to stop the string at.

    Returns:
    The string, stopped at the first occurrence of a word in the list.
    """

    for word in words:
        if word in string:
            return string.partition(word)[0]

    return string
# ```

# Here is an example of how to use the `stop_string_at_first_occurrence()` function:

# ```python
# string = "This is a string with the words 'stop' and 'word'."
# words = ["stop", "word"]

# print(stop_string_at_first_occurrence(string, words))
# ```

# This will print the following output to the console:

# ```
# This is a string with the words 'stop'.
# ```


def _build_index_config(embedding_gcs_uri: str, dimensions: int):
    _treeAhConfig = struct_pb2.Struct(
        fields={
            "leafNodeEmbeddingCount": struct_pb2.Value(number_value=500),
            "leafNodesToSearchPercent": struct_pb2.Value(number_value=7),
        }
    )
    _algorithmConfig = struct_pb2.Struct(
        fields={"treeAhConfig": struct_pb2.Value(struct_value=_treeAhConfig)}
    )
    _config = struct_pb2.Struct(
        fields={
            "dimensions": struct_pb2.Value(number_value=dimensions),
            "approximateNeighborsCount": struct_pb2.Value(number_value=150),
            "distanceMeasureType": struct_pb2.Value(string_value="DOT_PRODUCT_DISTANCE"),
            "algorithmConfig": struct_pb2.Value(struct_value=_algorithmConfig),
            "shardSize": struct_pb2.Value(string_value="SHARD_SIZE_SMALL"),
        }
    )
    metadata = struct_pb2.Struct(
        fields={
            "config": struct_pb2.Value(struct_value=_config),
            "contentsDeltaUri": struct_pb2.Value(string_value=embedding_gcs_uri),
        }
    )

    return metadata

map_prompt_template = """
    Write a concise summary of the following:

    {text}

    CONSCISE SUMMARY:
    """
map_prompt = PromptTemplate(
    template=map_prompt_template
    , input_variables=["text"]
)

combine_prompt_template = """
    Write a concise summary of the following:

    {text}

    CONSCISE SUMMARY IN BULLET POINTS:
    """
combine_prompt = PromptTemplate(
    template=combine_prompt_template
    , input_variables=["text"]
)


class ResourceNotExistException(Exception):
    def __init__(self, resource: str, message="Resource Does Not Exist."):
        self.resource = resource
        self.message = message
        super().__init__(self.message)

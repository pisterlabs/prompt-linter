# -*- coding: utf-8 -*-
"""常用的AI工具。使用前，需先使用"addAPIKey"添加OpenAI密鑰。可添加多個，並在exceed quota等情況時自動切換。"""
import os
import tiktoken
import numpy as np
from openai.embeddings_utils import cosine_similarity as _cosine_similarity
from openai.error import *
from utils.global_value_utils import *
from utils.classes.cross_module_enum import CrossModuleEnum
from utils.crypto_utils import getSHA256Hash_fromString
from utils.sqlite_utils import Database, NotFoundError
from typing import Iterable, Union


_enc = GetOrAddGlobalValue("_OPENAI_ENCODING", tiktoken.get_encoding('cl100k_base'))
_apiKeys: set = GetOrAddGlobalValue("_OPENAI_API_KEYS", set())

class NoAPIKeyError(Exception):
    pass
class ExceedTokenLimitError(Exception):
    pass

def errorHandler(func):
    '''
    處理OpenAI的error，如果是exceed quota，則自動切換API密鑰。
    '''
    def wrapper(*args, **kwargs):
        errorCount = 0
        while True:
            try:
                return func(*args, **kwargs)
            except (AuthenticationError, RateLimitError):
                # wrong api key
                if len(_apiKeys) > 0:
                    # try next api key
                    openai.api_key = _apiKeys.pop()
                else:
                    # no more api key
                    raise NoAPIKeyError('No more available API key. Please add more API key.')
            except OpenAIError as e:
                if errorCount < 3:
                    # other error, try again
                    errorCount += 1
                else:
                    # other error, no more try
                    raise e
    return wrapper

def addAPIKey(apiKey:Union[str, Iterable[str]]):
    '''
    添加API密鑰到一個全局的set。可添加多個，並在exceed quota等情況時自動切換。
    調用OpenAI的function都必須添加api。
    '''
    if isinstance(apiKey, str):
        apiKeys = [apiKey]
    else:
        apiKeys = tuple(apiKey)
    global _apiKeys
    for key in apiKeys:
        _apiKeys.add(key)
    if openai.api_key is None:
        openai.api_key = _apiKeys.pop()
if os.environ.get('OPENAI_API_KEY') is not None:
    addAPIKey(os.environ['OPENAI_API_KEY'])

def get_tokens(text:str)->list:
    '''Get tokens from text'''
    return _enc.encode(text)
def count_tokens(text:str)->int:
    '''Count tokens from text'''
    return len(get_tokens(text))

from .path_utils import EMBEDDING_CACHE_DB_PATH
_embedCacheDB = Database(EMBEDDING_CACHE_DB_PATH)
_embedCacheTable = _embedCacheDB.create_table('embedding_cache', pk=['hash', 'model'],
                                              columns={'hash': str, 'embedding': bytes, 'model': str},
                                              not_null=('hash', 'embedding'),
                                              if_not_exists=True)
class EmbedModel(CrossModuleEnum):

    OPENAI = ('openai', 1536) # name for sqlite, vector dimension
    # TODO: add more models

    @property
    def dimension(self)->int:
        return self.value[1]

DEFAULT_EMBED_MODEL = EmbedModel.OPENAI
'''YOu can change this to other model by setting environment variable "EMBEDDING_MODEL" to the name of the model in EmbedModel(Enum).'''
if os.getenv("EMBEDDING_MODEL", None) is not None:
    model_name = os.getenv("EMBEDDING_MODEL", None)
    for model in EmbedModel:
        if model.name.lower() == model_name.lower() or model.value[0].lower() == model_name.lower():
            DEFAULT_EMBED_MODEL = model
            break
DEFAULT_EMBED_DIMENSION = DEFAULT_EMBED_MODEL.dimension

@errorHandler
def get_embedding_vector(text:str, model=DEFAULT_EMBED_MODEL)->np.ndarray:
    '''Get embedding vector from text (through OpenAI API)'''
    hash = getSHA256Hash_fromString(text)
    model_name = model.value[0]
    try: # try to get from cache
        result = _embedCacheTable.get([hash, model_name])
        return np.frombuffer(result['embedding'], dtype=np.float32)
    except NotFoundError:
        if model == EmbedModel.OPENAI:
            v = openai.Embedding.create(input=[text], model='text-embedding-ada-002')['data'][0]['embedding']
            embed = np.array(v, dtype=np.float32)
            _embedCacheTable.insert({'hash': hash, 'embedding': embed.tobytes(), 'model': model_name})
        # TODO: add more models
        return embed

class ChatModel(CrossModuleEnum):
    GPT3_5 = ('gpt-3.5-turbo', 4096)
    GPT3_5_16K = ('gpt-3.5-turbo-16k', 16384)
    GPT4 = ('gpt-4', 8192)
    GPT4_32K = ('gpt-4-32k', 32768)

@errorHandler
def get_chat(messages, roles=None, model:ChatModel=ChatModel.GPT3_5, temperature=0.5, frequency_penalty=0, presence_penalty=0,
             autoUpgradeModel=True, maxTokenRatio=0.75, timeout=5)->str:
    '''
    :param messages: list of messages(if only 1, then can be str), e.g. ["Hello, who are you?", "I am doing great. How about you?"]
    :param roles: list of roles(if only 1, then can be str), e.g. ["system", "assistant"], if None, then all roles are "user"
    :param temperature: higher->more random, lower->more deterministic [0, 2]
    :param frequency_penalty: higher->less repetition, lower->more repetition [-2, 2]
    :param presence_penalty: higher->encourage model to talk about new topics, lower->encourage model to repeat itself [-2, 2]
    :param autoUpgradeModel: if exceed token limit, then automatically upgrade model to a larger one (e.g. GPT-3.5 -> GPT-3.5-16K)
    :param maxTokenRatio: maximum tokens allowed for messages (the rest is for response)
    '''
    if isinstance(roles, str):
        roles = [roles]
    if isinstance(messages, str):
        messages = [messages]
    if roles is None:
        roles = ['user'] * len(messages)
    elif len(roles) != len(messages):
        raise ValueError('roles and messages must have the same length')
    modelName, tokenLimit = model.value
    tokenCount = count_tokens(' '.join(messages))
    tokenNumAllowed = int(tokenLimit * maxTokenRatio)
    if tokenCount > tokenNumAllowed:
        if not autoUpgradeModel or model in (ChatModel.GPT3_5_16K, ChatModel.GPT4_32K):
            raise ExceedTokenLimitError(f'Exceed token limit ({tokenNumAllowed} tokens allowed, {tokenCount} tokens given)')
        else:
            if model == ChatModel.GPT3_5:
                modelName, tokenLimit = ChatModel.GPT3_5_16K.value
            elif model == ChatModel.GPT4:
                modelName, tokenLimit = ChatModel.GPT4_32K.value
            tokenNumAllowed = int(tokenLimit * maxTokenRatio)
            if tokenCount > tokenNumAllowed:
                raise ExceedTokenLimitError(f'Still exceed token limit ({tokenNumAllowed} tokens allowed, {tokenCount} tokens given) after upgrading model')

    return openai.ChatCompletion.create(
        model=modelName,
        messages=[{"role": role, "content": message} for role, message in zip(roles, messages)],
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        timeout=timeout,
    )['choices'][0]['message']['content']


def normalize(vector):
    return vector / np.linalg.norm(vector)

def cosine_similarity(v1, v2):
    '''調用openai.embeddings_utils.cosine_similarity'''
    if isinstance(v1, list):
        v1 = np.array(v1)
    if isinstance(v2, list):
        v2 = np.array(v2)
    v1.reshape(1, -1)
    v2.reshape(-1, 1)
    return _cosine_similarity(v1, v2)



__all__ = ['addAPIKey', 'get_tokens', 'count_tokens', 'get_embedding_vector', 'get_chat', 'normalize', 'cosine_similarity', 'ChatModel', 'EmbedModel']

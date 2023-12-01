import openai
import tiktoken

from .model import Model

cached_model_details = None


class OllamaModel(Model):
    def __init__(self, name):

from components.chat_openai_connector import ChatOpenAIConnector
from components.huggingface_connector import HuggingFaceConnector
from components.openai_davinci_connector import OpenAIDavinciConnector
from config import LLM_MODEL_NAME


class LLMConnectorFactory:
    """
    A factory class for creating language model connectors.

    This class provides a static method for creating a connector based on the LLM_MODEL_NAME configuration variable.
    Depending on the model name specified in the configuration:
    - If the LLM_MODEL_NAME starts with 'text-davinci', an OpenAIDavinciConnector is returned.
    - If the LLM_MODEL_NAME starts with 'flan-t5', a HuggingFaceConnector is returned.
    - If the LLM_MODEL_NAME starts with 'gpt-3.5-turbo', a ChatOpenAIConnector is returned.
    Otherwise, a ValueError is raised.

    Example usage:
    >>> connector = LLMConnectorFactory.create_connector()
    """

    @staticmethod
    def create_connector():
        if LLM_MODEL_NAME.startswith('text-davinci'):
            return OpenAIDavinciConnector()
        elif LLM_MODEL_NAME.startswith('flan-t5'):
            return HuggingFaceConnector()
        elif LLM_MODEL_NAME.startswith('gpt-3.5-turbo'):
            return ChatOpenAIConnector()
        # Add more connectors as required
        raise ValueError(f"Unsupported LLM model: {LLM_MODEL_NAME}")

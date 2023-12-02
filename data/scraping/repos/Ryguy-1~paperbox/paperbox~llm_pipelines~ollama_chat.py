from langchain.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class OllamaChat(object):
    """Defines an Ollama LangChain Chat Prompted Model."""

    def __init__(
        self,
        ollama_model_name: str,
    ) -> None:
        """
        Initialize the Ollama Markdown Writer.

        Params:
            ollama_model_name (str): The Ollama model name to use.
        """
        self.ollama_model_name = ollama_model_name
        self.llm = Ollama(
            model=self.ollama_model_name,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.8,
        )

    def chat(self, chat: str) -> str:
        """
        Chat with Ollama.

        Params:
            chat (str): The chat to send to Ollama.

        Streams:
            stdout: The Ollama response.
        """
        self.llm(prompt=chat)
        print("\n")

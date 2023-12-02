from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

ollama = Ollama(
                model="dolphin2.2-mistral",
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                )

print(ollama("Why is the sky blue?"))

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()

mistral_path = os.getenv("MISTRAL_PATH")

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=mistral_path,
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True, # Verbose is required to pass to the callback manager
)

prompt = """
Question: A rap battle between Stephen Colbert and John Oliver.
Resp:

"""
llm(prompt)
# https://github.com/facebookresearch/llama-recipes/blob/main/demo_apps/HelloLlamaLocal.ipynb

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# for token-wise streaming so you'll see the answer gets generated token by token when Llama is answering your question
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="./llama-2-7b-chat.Q8_0.gguf", # Download from Huggable Face - use TheBloke
    temperature=0.0,
    top_p=1,
    n_ctx=6000,
    callback_manager=callback_manager, 
    verbose=True,
)

question = "Hello world!"
answer = llm(question)

# Alternatively, you can use LangChain's PromptTemplate for some flexibility in your prompts and questions.
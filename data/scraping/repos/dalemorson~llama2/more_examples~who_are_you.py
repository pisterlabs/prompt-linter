from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Add additional imports
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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

prompt = PromptTemplate.from_template(
    "What is {what}?"
)
chain = LLMChain(llm=llm, prompt=prompt)
answer = chain.run("llama2")

# Response:
# LLaMA (LLAMA) is an open-source, transformer-based language model developed by Facebook AI Research (FAIR). 
# It is designed to be a flexible and efficient alternative to other large language models like BERT and RoBERTa. 
# LLaMA can be fine-tuned on specific tasks like question answering, sentiment analysis, and text classification, making it a versatile tool for natural language processing (NLP) tasks.
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="C:/Users/stahl/Downloads/llm/llama.cpp/wizard-vicuna-13B.ggmlv3.q4_0.bin", callback_manager=callback_manager, verbose=True)


llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?, Let's think step by step."

llm_chain.run(question)
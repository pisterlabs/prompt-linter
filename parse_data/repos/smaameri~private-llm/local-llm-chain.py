from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """
You are a friendly chatbot assistant that responds in a conversational manner to users questions. Keep the
answers short, unless specifically asked by the user to elaborate on something. Don't make your answers too
technical, unless specifically asked to. Keep them light.

Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = GPT4All(
    model='./models/ggml-gpt4all-j-v1.3-groovy.bin',
    callbacks=[StreamingStdOutCallbackHandler()]
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

query = input("Prompt: ")
llm_chain(query)
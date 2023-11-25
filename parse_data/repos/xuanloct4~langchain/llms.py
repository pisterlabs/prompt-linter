
import environment
import os

def CohereLLM():
    from langchain.llms import Cohere
    llm = Cohere(cohere_api_key=os.environ.get("COHERE_API_KEY"))
    return llm

def AI21LLM():
    # !pip install ai21
    ## get AI21_API_KEY. Use https://studio.ai21.com/account/account
    # from getpass import getpass
    # AI21_API_KEY  = getpass()
    from langchain.llms import AI21
    llm = AI21(ai21_api_key=os.environ.get("AI21_API_KEY"))

    from langchain import PromptTemplate, LLMChain
    # template = """Question: {question}

    # Answer: Let's think step by step."""

    # prompt = PromptTemplate(template=template, input_variables=["question"])
    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    # question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

    # print(llm_chain.run(question))
    return llm

def StochasticAILLM():
    # from getpass import getpass
    # STOCHASTICAI_API_KEY = getpass()
    # import os
    # os.environ["STOCHASTICAI_API_KEY"] = STOCHASTICAI_API_KEY
    # YOUR_API_URL = getpass()
    from langchain.llms import StochasticAI
    from langchain import PromptTemplate, LLMChain
    YOUR_API_URL="https://api.stochastic.ai/v1/modelApi/submit/gpt-j"
    # YOUR_API_URL="https://api.stochastic.ai/v1/modelApi/submit/flan-t5"
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = StochasticAI(api_url=YOUR_API_URL)
    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    # question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

    # print(llm_chain.run(question))
    return llm


def GooseAILLM():
    from langchain.llms import GooseAI
    llm = GooseAI()
    return llm

def OpenAILLM():
    from langchain import OpenAI
    llm = OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=1000)
    return llm
def ChatLLMOpenAI():
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) 
    return llm

def GPT4AllLLM():
    from langchain.llms import GPT4All
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    callbacks = [StreamingStdOutCallbackHandler()]
    local_path = '../gpt4all/chat/ggml-gpt4all-l13b-snoozy.bin' 
    llm = GPT4All(model=local_path, n_ctx=2048, callbacks=callbacks, verbose=True)
    return llm

def defaultLLM():
    llm = OpenAILLM()
    # llm = CohereLLM()
    # llm = GPT4AllLLM()
    # llm = AI21LLM()
    # llm = StochasticAILLM()
    return llm


def defaultChatLLM():
    llm = ChatLLMOpenAI()
    return llm

defaultChatLLM = defaultChatLLM()
defaultLLM = defaultLLM()
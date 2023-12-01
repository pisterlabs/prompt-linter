from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.utilities import SerpAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.agents import AgentType
import streamlit as st


@st.cache_resource
def load__internet_model(model_architecture, model_name, model_path, max_token, temp, top_p, top_k):
    
    local_path = '{}/{}'.format(model_path, model_name)  # replace with your desired local file path
    callbacks = CallbackManager([StreamingStdOutCallbackHandler()])
    
    if model_architecture == "GPT4ALL":
        model = GPT4All(model=local_path, verbose=True, callbacks=callbacks,  n_predict=max_token, temp=temp, top_p=top_p, top_k=top_k)
    if model_architecture == "Llama-cpp":
        model = LlamaCpp(model_path=local_path, verbose=True, callbacks=callbacks, max_tokens=max_token,temperature=temp,top_p=top_p,top_k=top_k)

    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
    ]    


    template = """The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=template
    )

    memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history")
    
    agent_chain = initialize_agent(tools=tools, llm=model, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

    return agent_chain


@st.cache_resource
def load_model(model_architecture, model_name, model_path, max_token, temp, top_p, top_k):
    
    local_path = '{}/{}'.format(model_path, model_name)  # replace with your desired local file path
    callbacks = CallbackManager([StreamingStdOutCallbackHandler()])
    
    if model_architecture == "GPT4ALL":
        model = GPT4All(model=local_path, verbose=False, n_predict=max_token, temp=temp, top_p=top_p, top_k=top_k)
    if model_architecture == "Llama-cpp":
        model = LlamaCpp(model_path=local_path, callback_manager=callbacks, verbose=True, max_tokens=max_token,temperature=temp,top_p=top_p,top_k=top_k)
    
    template = """
            The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.
            
            Current conversation:
            {history}
            Human: {input}
            AI Assistant:"""
    
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=template
    )
    
    conversation = ConversationChain(
    llm=model, 
    verbose=True, 
    memory=ConversationBufferWindowMemory(k=10),
    callbacks=callbacks,
    prompt=PROMPT
    )

    return conversation


class WanderingBrain:

    def __init__(self) -> None:
        pass


    def run_model(self, model_architecture, model_name, prompt, model_path, max_token, temp, top_p, top_k,is_internet=False):
        
        if model_architecture == "GPT4ALL":
            try:
                agent_chain = load_model(model_architecture=model_architecture, model_name=model_name, model_path=model_path[0], max_token=max_token, temp=temp, top_p=top_p, top_k=top_k)
            except:
                agent_chain = load_model(model_architecture=model_architecture, model_name=model_name, model_path=model_path[1], max_token=max_token, temp=temp, top_p=top_p, top_k=top_k)
        
        if model_architecture == "Llama-cpp":
            try:
                agent_chain = load_model(model_architecture=model_architecture, model_name=model_name, model_path=model_path[0], max_token=max_token, temp=temp, top_p=top_p, top_k=top_k)
            except:
                agent_chain = load_model(model_architecture=model_architecture, model_name=model_name, model_path=model_path[1], max_token=max_token, temp=temp, top_p=top_p, top_k=top_k)
        
        
        if is_internet == False:
            return agent_chain.predict(input=prompt)
        else:
            return agent_chain.predict(input=prompt)



import os
from langchain import PromptTemplate, LLMChain, HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

# initialize Hub LLM
hub_llm = HuggingFaceHub(
                         repo_id='bigscience/bloom',
                         model_kwargs={'temperature':1e-10}
)

conversation = ConversationChain(
    llm=hub_llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

conversation.predict(input="Hi there!")


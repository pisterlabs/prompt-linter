from PyQt5.QtCore import pyqtSignal, QThread
from langchain import SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool


def get_agent():
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = ChatOpenAI(temperature=0, model_name='gpt-4')
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
    ]
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
    )


class CatGPTAgent:
    def __init__(self):
        self.agent = get_agent()

    def get_response(self, message):
        return self.agent.run(message)


class CatResponseThread(QThread):
    response_signal = pyqtSignal(str)

    def __init__(self, cat_gpt, text):
        super().__init__()
        self.cat_gpt = cat_gpt
        self.text = text

    def run(self):
        cat_response = self.cat_gpt.get_response(self.text)
        self.response_signal.emit(cat_response)


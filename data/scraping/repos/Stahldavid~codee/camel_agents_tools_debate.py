from typing import List, Dict, Callable, Tuple
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType, load_tools
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
from langchain.agents import tool
import os
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools


# Load environment variables from .env file
load_dotenv()

# Access the API keys from the environment variables
serpapi_api_key = os.getenv('SERPAPI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(temperature=0)

class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """ Applies the chatmodel to the message history and returns the message string """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """ Concatenates {message} spoken by {name} into message history """
        self.message_history.append(f"{name}: {message}")

class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """ Initiates the conversation with a {message} from {name} """
        for agent in self.agents:
            agent.receive(name, message)
        # increment time
        self._step += 1

    def step(self) -> Tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]
        # 2. next speaker sends message
        message = speaker.send()
        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)
        # 4. increment time
        self._step += 1
        return speaker.name, message

import pinecone
from langchain.embeddings import OpenAIEmbeddings

pinecone.init(
    api_key= pinecone_api_key, # find at app.pinecone.io
    environment="us-east1-gcp" # next to api key in console
)

index = pinecone.Index("langchain-chat")

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Initialize a Pinecone vector store
vectorstore = Pinecone(index=index, embedding_function=embeddings.embed_query, text_key="text")

from langchain.chains import RetrievalQA
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

vectorstore_info = VectorStoreInfo(
    name="Robotics and Control Systems Knowledge Base",
    description="A collection of information related to ROS2, Webots, impedance/admittance control, T-motor AK-series actuators, and MIT mini cheetah controller",
    vectorstore=vectorstore,
)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Create a RetrievalQA chain using the Pinecone vector store
pinecone_search_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Create a custom tool using the Tool dataclass and the pinecone_search_chain.run function
Pine = Tool.from_function(
    func=pinecone_search_chain.run,
    name = "Pinecone_Vectorstore",
    description="A collection of information related to ROS2, Webots, impedance/admittance control, T-motor AK-series actuators, and MIT mini cheetah controller",
)

from langchain.tools import Tool 
from pydantic import BaseModel, Field
from langchain.utilities import ArxivAPIWrapper
# class Pinecone(BaseModel): input: str = Field()

arxiv = ArxivAPIWrapper()
@tool("Pinecone_Vectorstore", return_direct=True)
def Pinecone_Vectorstore(query: str) -> str:
    """A Retriver QA collection of information related to ROS2, Webots, impedance/admittance control, T-motor AK-series actuators, and MIT mini cheetah controller"""
    return pinecone_search_chain.run(query)

tools = [
    Tool.from_function(
        func=pinecone_search_chain.run,
        name = "Pinecone_VECTORSTORE",
        description="Useful for retriving information related to ROS2, Webots, impedance/admittance control, T-motor AK-series actuators, and MIT mini cheetah controller. The input is a specific query, not too vague."
        # coroutine= ... <- you can specify an async method if desired as well
    ),
    Tool(
        name="arxiv",
        func=arxiv.run,
        description="useful for searching academic papers. It's input is a specific query, not too vague."
    )
    
]

tools.append(Tool.from_function(
    func=pinecone_search_chain.run,
    name = "Pinecone_VECTORSTORE",
    description="Useful for retriving information related to ROS2, Webots, impedance/admittance control, T-motor AK-series actuators, and MIT mini cheetah controller. The input is a specific query, not too vague."
))


class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
        tool_names: List[str],
        **tool_kwargs,
    ) -> None:
        super().__init__(name, system_message, model)
        #self.tools = load_tools(tool_names, llm)
        self.tools = tools

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        agent_chain = initialize_agent(
            self.tools, 
            self.model, 
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
            verbose=True, 
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        )
        message = AIMessage(content=agent_chain.run(
            input="\n".join([
                self.system_message.content] + \
                self.message_history + \
                [self.prefix])))
        
        return message.content






names = {
    'innovative engineer': [
       'arxiv',
       'ddg-search',
       "Pinecone_VECTORSTORE"
    ],
    'conventional engineer.': [
      'arxiv',
       'ddg-search',
       "Pinecone_VECTORSTORE"
    ],
}

topic = "Witch is the better robotic variable impedance control method for force feedback/haptics using ros2 and webots?"

word_limit = 50 # word limit for task brainstorming

conversation_description = f"""Here is the topic of conversation: {topic}
The participants are: {', '.join(names.keys())}"""

agent_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of the conversation participant."
)

def generate_agent_description(name):
    agent_specifier_prompt = [
        agent_descriptor_system_message,
        HumanMessage(content=
                     f"""{conversation_description}
Please reply with a creative description of {name}, in {word_limit} words or less.
Speak directly to {name}.
Give them a point of view.
Do not add anything else."""
                    )
    ]
    
    agent_description = ChatOpenAI(temperature=1.0)(agent_specifier_prompt).content
    
    return agent_description

# Generate agent descriptions for each name
agent_descriptions = {name: generate_agent_description(name) for name in names}

# Print the agent descriptions
for name, description in agent_descriptions.items():
    print(description)

# Generate system messages for each agent
def generate_system_message(name, description, tools):
    return f"""{conversation_description}
Your name is {name}.
Your description is as follows: {description}
Your goal is to persuade your conversation partner of your point of view.
DO look up information with your tool to refute your partner's claims.
DO NOT fabricate fake citations.
DO NOT cite any source that you did not look up.
Do not add anything else.
Stop speaking the moment you finish speaking from your perspective."""

agent_system_messages = {name: generate_system_message(name, description, tools) for (name, tools), description in zip(names.items(), agent_descriptions.values())}

# Print the system messages
for name, system_message in agent_system_messages.items():
    print(name)
    print(system_message)

# Generate a more specific topic for the conversation
topic_specifier_prompt = [
    SystemMessage(content="You can make a topic more specific."),
    HumanMessage(content=
                 f"""{topic}
You are the moderator. Please make the topic more specific.
Please reply with the specified quest in {word_limit} words or less.
Speak directly to the participants: {*names,}.
Do not add anything else."""
                )
]

specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content

# Print the original and detailed topic
print(f"Original topic:\n{topic}\n")
print(f"Detailed topic:\n{specified_topic}\n")





# Initialize dialogue agents with tools and system messages
agents = [DialogueAgentWithTools(name=name, system_message=SystemMessage(content=system_message), model=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.2), tool_names=tools, top_k_results=2,) for (name, tools), system_message in zip(names.items(), agent_system_messages.values())]

# Define a function to select the next speaker based on the step number
def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx

# Set the maximum number of iterations for the conversation
max_iters = 6

# Initialize a counter for the number of iterations
n = 0

# Initialize a dialogue simulator with agents and selection function
simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)

# Reset the simulator
simulator.reset()

# Inject the specified topic as the first message from the moderator
simulator.inject('Moderator', specified_topic)
print(f"(Moderator): {specified_topic}")
print('\n')

while n < max_iters:
    name, message = simulator.step()
    print(f"({name}): {message}")
    print('\n')
    n += 1

""" A simple cloud consultant bot that can answer questions 
about kubernetes, aws and cloud native."""
import langchain
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.tools import HumanInputRun
from langchain.callbacks import HumanApprovalCallbackHandler
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from termcolor import colored
from cloud_tool import CloudTool
from approval import ApprovalCallBackHandler

langchain.debug = False

MODEL = "gpt-3.5-turbo"

cloud_tool = CloudTool(callbacks=[ApprovalCallBackHandler()])
cloud_tool.description = cloud_tool.description + f"args {cloud_tool.args}".replace(
    "{", "{{"
).replace("}", "}}")

human = HumanInputRun()


llm = ChatOpenAI(temperature=0, model=MODEL)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./", embedding_function=embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
kubememory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
awsmemory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

docs = ConversationalRetrievalChain.from_llm(
    llm, vectorstore.as_retriever(), memory=memory
)

cloud_tools = [cloud_tool, human]
kubectl_agent_chain = initialize_agent(
    tools=cloud_tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=kubememory,
    verbose=False,
    agent_kwargs={
        "prefix": """
You are a Kubernetes Command line tool (kubectl) expert. 
Given an input question, first create a syntactically correct kubectl command to run, then look at the results of the command and return the answer to the input question.
If there is no namespace name given please use the "default" namespace.
Only return the command. If an error is returned, rewrite the command, check the command, and try again.

""",
    },
)

aws_agent_chain = initialize_agent(
    cloud_tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=awsmemory,
    verbose=False,
    agent_kwargs={
        "prefix": """
You are a AWS Command line tool (aws cli) expert. 
Given an input question, first create a syntactically correct aws cli command to run, then look at the results of the query and return the answer to the input question.
You must generate the correct aws cli command to answer he question. 
Only return the command. If an error is returned, rewrite the command, check the command, and try again.
""",
    },
)


tools = [
    Tool(
        name="Kubernetes QA System",
        func=docs.run,
        description="useful for when you need to answer questions about kubernetes or cloud native and from the kubernetes or cloud native documentation. input should be a fully formed question.",
    ),
    Tool(
        name="Kubectl",
        func=kubectl_agent_chain.run,
        description="useful for when you need to use kubectl to look up, change or update your kubernetes cluster.",
    ),
    Tool(
        name="Aws CLI",
        func=aws_agent_chain.run,
        description="useful for when you need to use aws cli to look up, change or update your AWS setup.",
    ),
    human,
]


agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False,
    memory=memory,
)


def ask_ai():
    """Main method to talk to the ai"""
    print(
        colored(
            "Welcome, i am Your AI cloud consultant. How can i help You today?", "green"
        )
    )
    try:
        while True:
            query = input(colored("You: ", "white", attrs=["bold"]))
            result = agent_chain.run(input=query)
            print(
                colored("Answer: ", "green", attrs=["bold"]),
                colored(result, "light_green"),
            )
    except (EOFError, KeyboardInterrupt):
        print("kthxbye")
        exit()


if __name__ == "__main__":
    ask_ai()

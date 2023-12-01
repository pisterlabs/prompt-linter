from dotenv import load_dotenv
from langchain.agents.mrkl.prompt import PREFIX, SUFFIX
from langchain.chains.conversation.memory import \
    ConversationSummaryBufferMemory

from ask_architecture_questions import ask_architecture_knowledge

load_dotenv()

from langchain.agents import initialize_agent, load_tools
from langchain.llms import OpenAI, OpenAIChat

from tools import ask_for_approval, send_message

llm = OpenAIChat()

agent = initialize_agent(
    [
        ask_for_approval,
        ask_architecture_knowledge,
    ],
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    agent_kwargs={
        "prefix": "You are Jack's personal knowledge assistant. You will receive inputs and suggest responses on behalf of Jack. Ask Jack for approval before sending a message. Implement the feedback and ask again. Feedback can be asked N times. Send the message exactly, without quotes."
        + PREFIX,
        "suffix": "The final answer should be a message that you will send on behalf of Jack. The final answer should always be approved."
        + SUFFIX,
    },
    memory=ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=60,
    ),
)

while True:
    i = input("Person: ")
    send_message(agent.run(input=f'A person said "{i}"'))

from dotenv import load_dotenv
import re

load_dotenv()

# variable_pattern = r"\{([^}]+)\}"
# input_variables = re.findall(variable_pattern, template)

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris."),
]

prompt = HumanMessage(
    content="I'd like to know more about the city you just mentioned."
)

messages.append(prompt)

llm = ChatOpenAI(model="gpt-4")

response = llm(messages)

print(response)
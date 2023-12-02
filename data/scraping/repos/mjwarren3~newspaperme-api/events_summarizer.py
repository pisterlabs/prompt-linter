from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

chat_model = ChatOpenAI()

template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate {num_of_objects} objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""

human_template = "{category}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

summary_chain = chat_prompt | chat_model
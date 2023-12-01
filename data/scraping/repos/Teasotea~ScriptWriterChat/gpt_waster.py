import os
from langchain.chat_models import ChatOpenAI

from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from prompt_lib import STARTER_PROMPT

from dotenv import load_dotenv

load_dotenv()

if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
    print("OPENAI_API_KEY is not set")
    exit(1)
else:
    print("OPENAI_API_KEY is set")

chat = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
PROMPT = PromptTemplate(template=STARTER_PROMPT, input_variables=["input", "history"])
conversation = ConversationChain(prompt=PROMPT, llm=chat, memory=ConversationSummaryMemory(llm=chat), verbose=True)


def generate_response(prompt):
    return conversation.predict(input=prompt)


for i in range(250):
    response = generate_response(STARTER_PROMPT)
    print("Response:" + str(i))
print(response)

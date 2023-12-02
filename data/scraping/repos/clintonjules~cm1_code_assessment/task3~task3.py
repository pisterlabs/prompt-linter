import os

# Enter yuor OpenAI key
os.environ["OPENAI_API_KEY"] = 'YOUR_KEy'

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "The following is a friendly conversation between a human and an AI. The AI is talkative and "
        "provides lots of specific details from its context. If the AI does not know the answer to a "
        "question, it truthfully says it does not know."
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

print(conversation.predict(input="Hi there! My name is Clint"))

print(conversation.predict(input="Tell em a joke about my name"))

print(conversation.predict(input="What are the last 2 message we exchanged?"))

print("-------------SystemMessagePromptTemplate---------------\n")
print(prompt.messages[0])

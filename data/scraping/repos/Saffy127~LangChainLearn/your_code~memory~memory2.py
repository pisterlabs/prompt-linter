from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(
  llm=llm,
  verbose=True,
  memory=ConversationBufferMemory()
)

conversation.predict(input="Hi there!")


memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("Whats up?")

memory.load_variables({})

memory.load_memory_variables({})

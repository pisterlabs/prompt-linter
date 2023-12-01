from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

conversation.run("Hi there!")

output = conversation.run("I'm doing well! Just having a conversation with an AI.")
print(output)

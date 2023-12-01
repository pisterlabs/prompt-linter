from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


from langchain.callbacks import get_openai_callback

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result


load_dotenv()

llm = ChatOpenAI()

conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(), 
    verbose=True)

# print(conversation.prompt.template)

print("Hello! How can I help you?")


## Second part  

while True:
    user_input = input("> ")
    ai = count_tokens(conversation,user_input)
    print(ai)



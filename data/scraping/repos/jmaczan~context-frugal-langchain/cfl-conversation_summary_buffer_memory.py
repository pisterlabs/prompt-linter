from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
import dotenv
dotenv.load_dotenv()


llm = OpenAI(temperature=0)
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationSummaryBufferMemory(
        llm=OpenAI(), max_token_limit=30, verbose=True, return_messages=True),  # return_messages=True
    verbose=True
)
conversation_with_summary.predict(input="Hi, I am Jed. How are you?")
print(conversation_with_summary.memory)
print(conversation_with_summary.memory.moving_summary_buffer)

conversation_with_summary.predict(input="I'm not a Jed. I'm Bob")
print(conversation_with_summary.memory)
print(conversation_with_summary.memory.moving_summary_buffer)

conversation_with_summary.predict(input="Who am I?")
print(conversation_with_summary.memory)
print(conversation_with_summary.memory.moving_summary_buffer)

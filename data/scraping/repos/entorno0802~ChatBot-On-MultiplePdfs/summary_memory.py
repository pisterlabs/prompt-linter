from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(temperature=0)
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(llm=OpenAI()),
    verbose=True
)
conversation_with_summary.predict(input="Hi, what's up?")
conversation_with_summary.predict(input="Tell me more about it!")
conversation_with_summary.predict(
    input="Very cool -- what is the scope of the project?")
print(conversation_with_summary.memory.load_memory_variables({}))

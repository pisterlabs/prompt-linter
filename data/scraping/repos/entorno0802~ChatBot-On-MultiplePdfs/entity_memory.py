from langchain.llms import OpenAI
from langchain.memory import ConversationEntityMemory
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(temperature=0)
memory = ConversationEntityMemory(llm=llm)
_input = {"input": "Deven & Sam are working on a hackathon project"}
memory.load_memory_variables(_input)
memory.save_context(
    _input,
    {"output": " That sounds like a great project! What kind of project are they working on?"}
)

_input = {"input": "Susan is a accountant"}
memory.load_memory_variables(_input)
memory.save_context(_input, {"output": " That's a nice job"})
print(memory.entity_store)

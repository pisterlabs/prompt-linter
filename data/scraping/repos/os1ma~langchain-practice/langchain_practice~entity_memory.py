from langchain.llms import OpenAI
from langchain.memory import ConversationEntityMemory
from util import initialize

initialize()

llm = OpenAI(temperature=0)

memory = ConversationEntityMemory(llm=llm)
_input = {"input": "Deven & Sam are working on a hackathon project"}
result = memory.load_memory_variables(_input)
print(result)
memory.save_context(
    _input,
    {
        "output": " That sounds like a great project! What kind of project are they working on?"
    },
)
result = memory.load_memory_variables({"input": "who is Sam"})
print(result)

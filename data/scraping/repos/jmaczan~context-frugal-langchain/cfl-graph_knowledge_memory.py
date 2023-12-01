from langchain.memory import ConversationKGMemory
from langchain.llms import OpenAI
import dotenv
dotenv.load_dotenv()

llm = OpenAI(temperature=0, verbose=True)
memory = ConversationKGMemory(llm=llm,  return_messages=True, verbose=True)
memory.save_context({"input": "say hi to sam"}, {"output": "who is sam"})
memory.save_context({"input": "sam is an enemy"}, {"output": "okay"})
memory.save_context({"input": "sam is a friend"}, {"output": "okay, I will remember that sam is not a friend anymore. Sam is an enemy!"})
print(memory.load_memory_variables({"input": "who is sam"}))

print(memory.get_current_entities("who is sam?"))
# print(memory.get_knowledge_triplets("her favorite color is red"))
print(memory.load_memory_variables({"input": "is sam a friend or enemy?"}))

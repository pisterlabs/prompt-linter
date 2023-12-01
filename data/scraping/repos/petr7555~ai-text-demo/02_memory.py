import dotenv
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationSummaryMemory

# Load OPENAI_API_KEY
dotenv.load_dotenv()

print("Simple and most commonly used")
memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")
print(memory.load_memory_variables({}))

print("With sliding window")
memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})
print(memory.load_memory_variables({}))

print("Memory for longer conversations which creates a summary")
llm = OpenAI(temperature=0)
memory = ConversationSummaryMemory(llm=llm)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "im working on better docs for chatbots"},
                    {"output": "oh, that sounds like a lot of work"})
memory.save_context({"input": "yes, but it's worth the effort"}, {"output": "agreed, good docs are important!"})
print(memory.load_memory_variables({}))

print("Use token length to determine when to flush interactions (=create a summary)")
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10)
memory.save_context({"input": "hi"}, {"output": "whats up"})
print(memory.load_memory_variables({}))
memory.save_context({"input": "not much you"}, {"output": "not much"})
print(memory.load_memory_variables({}))

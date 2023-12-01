from langchain.memory import ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

memory = ConversationSummaryMemory(llm=model)

# Every time you call .save_context or push anything into memory it will call the model to once again re-summarize the conversation.
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context(
    {"input": "my name is jacob"},
    {"output": "Nice to meet you jacob! What can I do for you today?"},
)
memory.save_context(
    {"input": "I want to know who discovered the theory of relativity"},
    {"output": "Albert Einstein"},
)


# print(memory.load_memory_variables({}))

# Now, let's try to use it in a chain.

# from langchain.chains import LLMChain

# conversation_with_summary = LLMChain(
#     llm=model,
#     memory=memory,
#     verbose=True,
#     prompt=ChatPromptTemplate.from_template("{input}"),
# )

# result = conversation_with_summary.predict(
#     input="What are some other scientists who contributed to the theory or relativity?"
# )

# print(result)

from langchain.llms import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(temperature=0)
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
)
conversation_with_summary.predict(
    input="What are some other scientists who contributed to the theory or relativity?"
)

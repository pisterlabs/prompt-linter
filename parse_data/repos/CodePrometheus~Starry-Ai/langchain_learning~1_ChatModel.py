from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# # ============================ Chat Models ============================ #

chat = ChatOpenAI(
    temperature=0,
    streaming=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True
)

# 1 Chat Messages
messages = [
    SystemMessage(content="Say the opposite of what the user says"),
    HumanMessage(content="I love programming.")
]
chat(messages)

# 2 支持多条消息作为输入
batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate this sentence from English to French. I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate this sentence from English to French. I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)
print(result)
print(result.llm_output)

# 3 Prompt Templates
from langchain.schema import SystemMessage
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

prompt = PromptTemplate(
    template="Propose creative ways to incorporate {food_1} and {food_2} in the cuisine of the users choice.",
    input_variables=["food_1", "food_2"]
)

system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
# Output of system_message_prompt
system_message_prompt.format(food_1="Bacon", food_2="Shrimp")
SystemMessage(content='Propose creative ways to incorportate Bacon and Shrimp in the cuisine of the users choice.',
              additional_kwargs={})
# Make HumanMessagePromptTemplate
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
# Create ChatPromptTemplate: Combine System + Human
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chat_prompt_with_values = chat_prompt.format_prompt(food_1="Bacon", \
                                                    food_2="Shrimp", \
                                                    text="I really like food from Germany.")
print(chat_prompt_with_values.to_messages())
response = chat(chat_prompt_with_values.to_messages()).content
print(response)

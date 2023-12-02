# Just follow alongs from quickstart guide - https://python.langchain.com/docs/get_started/quickstart
from langchain.llms import OpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

openai_api_key="..."
# LLMs
llm = OpenAI(openai_api_key=openai_api_key, temperature=0.9)
# print(llm.predict("What would be a good company name for company that makes colorful socks?"))

# Chat models
chat = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
# result = chat.predict_messages([HumanMessage(content="Translate this sentence from English to Chinese and Japanese. I love programming.")])

# print(result)

# Prompt templates with Chains LLM
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
# print(prompt.format(product="colorful socks"))

chain = LLMChain(llm=llm, prompt=prompt)
# print(chain.run("colorful socks"))

# Prompt templates with Chains Chat models
template = "You are a helpful assistant that translate {input_language} to {output_language}, including any available Romanizations."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
# print(chat_prompt.format_messages(input_language="English", output_language="Japanese", text="I love programming."))

chain = LLMChain(llm=chat, prompt=chat_prompt)
# print(chain.run(input_language="English", output_language="Chinese", text="I love programming."))

# Agents LLM

llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# agent.run("Give me a the highest temperature recorded wikipedia. What is that number raised to the 0.023 power?")

# Agents Chat models
chat = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# agent.run("Find a formula created by Einstein and use that with example values from wikipedia.")


# Memory LLM
from langchain import ConversationChain

llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)
# print(conversation.run("Hi there!"))
# print(conversation.run("Nice to meet you too! Just testing how to use Langchain!"))

# Memory Chat models
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "The AI is rude and funny at the same time, and talk back with excitement."
        "If the AI does not know the ansewr to a "
        "questions, it truthfully lies about it, and tells you why it did so."
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.9)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
print(conversation.predict(input="Hello my friend!"))
print(conversation.predict(input="What'ya upto?"))
print(conversation.predict(input="Me just learning more about AI and coding a bit with it, and see where I'd end up:)"))

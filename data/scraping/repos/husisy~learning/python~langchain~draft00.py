import os
import dotenv
import openai
import langchain

dotenv.load_dotenv()
assert 'SERPAPI_API_KEY' in os.environ
assert 'OPENAI_API_KEY' in os.environ
SERPAPI_API_KEY = os.environ['SERPAPI_API_KEY'] #not necessary
OPENAI_API_KEY = os.environ['OPENAI_API_KEY'] #not necessary
# openai.api_key = os.environ["OPENAI_API_KEY"]

openai.api_key = os.environ['OPENAI']

## llm
model = langchain.llms.OpenAI(temperature=0.9, openai_api_key=OPENAI_API_KEY, model='text-davinci-003')
x0 = model.predict('What would be a good company name for a company that makes colorful socks?')
# '\n\nSocktastic Colors.'


## llm+prompt
prompt = langchain.prompts.PromptTemplate.from_template('What is a good name for a company that makes {product}?')
prompt.format(product='colorful socks')
# 'What is a good name for a company that makes colorful socks?'
model = langchain.llms.OpenAI(temperature=0.9, openai_api_key=OPENAI_API_KEY, model='text-davinci-003')
chain = langchain.chains.LLMChain(llm=model, prompt=prompt)
x0 = chain.run('colorful socks')
# '\n\nRainbow Socks Company'


## chat
model = langchain.chat_models.ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
langchain.schema.HumanMessage
langchain.schema.AIMessage
langchain.schema.SystemMessage
langchain.schema.ChatMessage
x0 = model.predict_messages([langchain.schema.HumanMessage(content='Translate this sentence from English to French. I love programming.')])
# AIMessage(content='\n\nJe adore la programmation.', additional_kwargs={}, example=False)
x0.content
# '\n\nJe adore la programmation.'
x0 = model.predict('Translate this sentence from English to French. I love programming.')
# "\n\nJ'adore la programmation."


## chat+prompt
tmp0 = langchain.prompts.chat.SystemMessagePromptTemplate.from_template('You are a helpful assistant that translates {input_language} to {output_language}.')
tmp1 = langchain.prompts.chat.HumanMessagePromptTemplate.from_template('{text}')
chat_prompt = langchain.prompts.chat.ChatPromptTemplate.from_messages([tmp0, tmp1])
chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
# [SystemMessage(content='You are a helpful assistant that translates English to French.', additional_kwargs={}),
#  HumanMessage(content='I love programming.', additional_kwargs={}, example=False)]


## agent
# The language model we're going to use to control the agent.
model = langchain.llms.OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model='text-davinci-003')
tools = langchain.agents.load_tools(["serpapi", "llm-math"], llm=model, serpapi_api_key=SERPAPI_API_KEY) #'llm-math' tool require an LLM
agent = langchain.agents.initialize_agent(tools, model, agent=langchain.agents.AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
x0 = agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")
# x0='1.0967325284643423'

# > Entering new AgentExecutor chain...
#  I need to find the temperature first, then use the calculator to raise it to the .023 power.
# Action: Search
# Action Input: "High temperature in SF yesterday"
# Observation: High: 55.4ºf @1:30 AM Low: 53.6ºf @3:35 AM Approx. Precipitation / Rain Total: in. 1hr.
# Thought: I now have the temperature, so I can use the calculator to raise it to the .023 power.
# Action: Calculator
# Action Input: 55.4^.023
# Observation: Answer: 1.0967325284643423
# Thought: I now know the final answer.
# Final Answer: 1.0967325284643423
# > Finished chain.


## chat+memory
model = langchain.llms.OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model='text-davinci-003')
conversation = langchain.ConversationChain(llm=model, verbose=True)
x0 = conversation.run("Hi there!")
# x0 = " Hi there! It's nice to meet you. How can I help you today?"
x0 = conversation.run("I'm doing well! Just having a conversation with an AI.")
# x0 = " That's great! It's always nice to have a conversation with someone new. What would you like to talk about?"
# > Entering new ConversationChain chain...
# Prompt after formatting:
# The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
# Current conversation:
# Human: Hi there!
# AI:  Hi there! It's nice to meet you. How can I help you today?
# Human: I'm doing well! Just having a conversation with an AI.
# AI:
# > Finished chain.

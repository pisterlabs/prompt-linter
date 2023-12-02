from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chains import ConversationChain, LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def predict(llm):
  text = "What would be a good company name for a company that makes colorful socks?"
  print(llm(text)) # -> "Bright Sox." (or something similar)

def predict_with_chain(llm):
  prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
  )

  print(prompt.format(product="colorful socks")) # -> "What is a good name for a company that makes colorful socks?"

  chain = LLMChain(llm=llm, prompt=prompt)

  # Predict a company name running the chain
  print(chain.run(product="colorful socks")) # -> "Bright Sox." (or something similar)

def agent(llm):
  llm.temperature = 0

  # Load tools
  tools = load_tools(["serpapi", "llm-math"], llm=llm)

  agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

  agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")

def memory(llm):
  conversation = ConversationChain(llm=llm, verbose=True)

  output = conversation.predict(input="Hi there!")
  print(output)

  output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
  print(output)

def chat(llm):
  template = "You are a helpful assistant that translates {input_language} to {output_language}."
  system_message_prompt = SystemMessagePromptTemplate.from_template(template)
  human_template = "{text}"
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

  chain = LLMChain(llm=llm, prompt=chat_prompt)
  print(chain.run(input_language="English", output_language="Português", text="I love programming."))
  # -> "Eu amo programação."

if __name__ == "__main__":
  # Create a new language model with a high temperature for more creative results
  llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.9)

  # Create a new language model using HuggingFace -> Running locally
  hf_llm = HuggingFacePipeline.from_model_id(
    "gpt2",
    task="text-generation",
    pipeline_kwargs={ "max_new_tokens": 100 }
  )

  # Create a new language model using OpenAI GPT-3
  chat_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
  
  # Predict a company name
  predict(llm)

  # Predict a company name using a chain
  predict_with_chain(llm)

  # Create an agent and use it with some tools
  agent(llm)

  # Create a conversation chain
  memory(hf_llm)

  # Chat with an AI
  chat(chat_llm)

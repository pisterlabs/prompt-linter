import os
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities import SerpAPIWrapper
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.prompts import PromptTemplate
# import LLM:
from langchain.chat_models.openai import ChatOpenAI

from typing import Dict, List
from pydantic import BaseModel

class Generation(BaseModel):
    text: str
    generation_info: str = None

class TokenUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class LlmOutput(BaseModel):
    token_usage: TokenUsage
    model_name: str

class Output(BaseModel):
    generations: List[List[Generation]]
    llm_output: LlmOutput

# llm_babbage = OpenAI(temperature=0, model_name="babbage", request_timeout=30, max_retries=2)
# llm_curie = OpenAI(temperature=0, model_name="curie", request_timeout=30, max_retries=2)
# llm_davinci = OpenAI(temperature=0, model_name="davinci", request_timeout=30, max_retries=2)

# llm_gpt4 = OpenAI(temperature=0, model_name="gpt4", request_timeout=30, max_retries=2, verbose=True)
llm_gpt3_turbo = OpenAI(temperature=0, model_name="gpt-3.5-turbo", request_timeout=30, max_retries=2, verbose=True)

async def get_simple_response(input:str) -> Output:
  return await llm_gpt3_turbo.agenerate([input],)

# TODO: Use agents to do this call and parsing, and combine these two prompts:

async def is_asking_for_smart_mode(input:str):
  
  prompt = PromptTemplate(
    input_variables=["input"],
    template="""Determine the following input contains explicit requests like increased intelligence, extra thinking, gpt4, expensiveness, slowness, etc. If so, return "smart_mode: yes". If the input is not explicitly requesting increased intelligence, slowness, gpt4, your answer should be "smart_mode: no". ONLY write "smart_mode: yes" or "smart_mode: no". 

Examples:
<!begin_input> Hey Chatterbot, I am gonna need you to think real hard about this one! No need to be creative since I'm just gonna talk about code. <!end_input> 
smart_mode: yes

<!begin_input> Hey Chatterbot, let's brainstorm some funny song titles! <!end_input> 
smart_mode: no

<!begin_input> Help me code. <!end_input> 
smart_mode: no

<!begin_input> {input} <!end_input>
""")
  
  query = prompt.format(input=input)
  print("About to ask GPT 3 about: ", query)
  
  try:
    response : Output = await get_simple_response(query)
    response = response.generations[0][0].text
    response = response.split("smart_mode: ")[1]
    response = response.strip().lower()
    return response == "yes"
  except Exception as e:
    print("Error in is_asking_for_smart_mode", e)
    return False

# TODO: Combine with prompt above
async def get_recommended_temperature(input:str, default_temperature=0.3):
  
  prompt = PromptTemplate(
    input_variables=["input", "default_temperature"],
    template="""Please indicate the appropriate temperature for the LLM to respond to the following message, using a scale from 0.00 to 1.00. For tasks that require maximum precision, such as coding, please use a temperature of 0. For tasks that require more creativity, such as generating imaginative responses, use a temperature of 0.7-1.0. If an explicit temperature/creativity is requested, use that. (Remember to convert percentages to a range between 0 and 1.0) If the appropriate temperature is unclear, please use a default of {default_temperature}. Please note that the temperature should be selected based solely on the nature of the task, and should not be influenced by the complexity or sophistication of the message.

Examples:
<!begin_input> Get as creative as possible for this one! <!end_input>
temperature: 1.00

<!begin_input> Tell me a bedtime story about a dinosaur! <!end_input>
temperature: 0.80

<!begin_input> Let's write some code. (Be really smart please) <!end_input>
temperature: 0.00

<!begin_input> Temperature:88%
Model: Super duper smart! <!end_input>
temperature: 0.88

<!begin_input> How are you doing today? <!end_input>
temperature: {default_temperature}

###

<!begin_input>: {input} <!end_input>
""")
  
  query = prompt.format(default_temperature=default_temperature, input=input)
  print("About to ask GPT 3 about: ", query)
  
  try:
    response : Output = await get_simple_response(query)
    response = response.generations[0][0].text
    print("response: ", response)
    response = response.split("temperature: ")[1]
    response = response.strip().lower()
    # try to parse the response as a float:
    try:
      return float(response)
    except:
      return default_temperature
  except Exception as e:
    print("Error in is_asking_for_smart_mode", e)
    return default_temperature

def get_conversational_agent(model_name="gpt-3.5-turbo"):
  search = GoogleSerperAPIWrapper()
  # TODO: File a PR to fix this return_messages=True thing
  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
  search = SerpAPIWrapper()
  tools = [
      Tool(
          name = "dinosaurs",
          func=search.run,
          description="(Useful to learn about dinosaurs)."
      ),
      Tool(
          name = "search",
          func=search.run,
          description="(ONLY if your confidence in your answer is below 0.2, use this tool to search for information)"
      ),
  ]
  llm=ChatOpenAI(temperature=0, model=model_name, verbose=True, request_timeout=30, max_retries=0)
  agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory, request_timeout=30)
  return agent_chain


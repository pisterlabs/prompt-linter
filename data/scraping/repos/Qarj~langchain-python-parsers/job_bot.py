from dotenv import load_dotenv
load_dotenv()

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from Tools.JobSearchTool import JobSearchTool

from Journey.JobBot import build_formatted_journey_prompt

from ThreeUnderscoreParser.ThreeUnderscorePromptTemplate import ThreeUnderscorePromptTemplate
from ThreeUnderscoreParser.ThreeUnderscoreOutputParser import ThreeUnderscoreOutputParser

from JsonParser.JsonPromptTemplate import JsonPromptTemplate
from JsonParser.JsonOutputParser import JsonOutputParser

from langchain.agents import AgentType
from langchain.agents import initialize_agent


def run_custom_response_parser(prompt, output_parser):
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    result = agent_executor.run(journey_prompt)
    return result

llm = ChatOpenAI(temperature=0, model="gpt-4")

search = JobSearchTool()
tools = [
    Tool(
        name = search.name,
        func=search.run,
        description=search.description
    )
]

#
# Step 1: Build the journey prompt
#
# The journey prompt defines WHAT type of response we want from the agent
#

journey_prompt = build_formatted_journey_prompt()


#
# Step 2: Choose which response prompt template and response output parser to use
# 
# The response prompt template defines HOW the agent will respond
# 

#
# Step 2, Option 1: Use the built-in response prompt template and response output parser
#

agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
result = agent.run(journey_prompt)

#
# Step 2, Option 2: Use the custom Three Underscore response prompt template and response output parser
# 

# prompt = ThreeUnderscorePromptTemplate(tools=tools, input_variables=["input", "intermediate_steps"])
# output_parser = ThreeUnderscoreOutputParser()
# result = run_custom_response_parser(prompt, output_parser)

#
# Step 2, Option 3: Use the custom JSON response prompt template and response output parser
# 

# prompt = JsonPromptTemplate(tools=tools, input_variables=["input", "intermediate_steps"])
# output_parser = JsonOutputParser()
# result = run_custom_response_parser(prompt, output_parser)

#
# Step 3: Print the parsed response
#

print("\n\n")

print("JobBot: ", result)


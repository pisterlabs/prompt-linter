from dotenv import load_dotenv, find_dotenv
import openai
import os

import langchain
import re
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
from langchain.memory import ConversationBufferWindowMemory
from langchain import OpenAI, LLMChain
from langchain.tools import DuckDuckGoSearchRun

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate


# read local .env file
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

##################### CUSTOM AGENT #######################################
# Bot Medico para resolver dudas sobre pacientes y sintomas.             #
# Ejercicio para saber como crear un agente con memoria usando langchain.#
##########################################################################


# Definimos las herramientas que el agente usará.
search = DuckDuckGoSearchRun()

#Definimos una herramienta de busqueda mas precisa.
def duck_wrapper(input_text):
    search_result = search.run(f'site:webmd.com {input_text}')
    return search_result

tools = [
    Tool(
      name='Search WebMD',
        func=duck_wrapper,
        description='Busqueda mas especifica en una web medica.'
    )
]

# Creamos el prompt template, en ingles para que la API lo entienda mejor.
# Base template
template = '''
Answer the following questions as best as you can, bus speaking as compasionate medical professional. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now the final answer
Final Answer : the final answer to the original input question

Begin! Remember to answer as a compasionate medical professional when giving your final answer.

Previous conversation history:
{history}

Question: {input}
{agent_scratchpad}
'''

# Clase para el Set up del template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Se encarga de los 'intermediate steps' (AgentAction, Observation tuples) y los formatea de una manera particular.
        intermediate_steps = kwargs.pop('intermediate_steps')
        thoughts = ''
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the argent_scratchpad variable to that value
        kwargs['agent_scratchpad'] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs['tools'] = '\n'.join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs['tool_names'] = ', '.join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt_with_history = CustomPromptTemplate(
    template=template,
    tools=tools,
    # Omite las variables 'agents_scratchpad', 'tools' and 'tools_name' porque son generadas dinamicamente en la función superior.
    # Incluir las variables 'input' e 'intermediate_steps' porque son necesarias.
    input_variables=['input', 'intermediate_steps', 'history']
)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check si ha terminado el agente
        if 'Final Answer: ' in llm_output:
            return AgentFinish(
                #Return de valor en diccionario con key 'output'
                return_values={'output': llm_output.split('Final Answer:')[-1].strip()},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: ´{llm_output}´")
        action = match.group(1).strip()
        action_input = match.group(2)

        # Return de action y action_input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()

# Setup agent
llm = OpenAI(temperature=0)
# LLMChain consiste en LLM y un prompt
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)

tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation"],
    allowed_tools=tool_names
)

memory = ConversationBufferWindowMemory(k=5)

# Un Agent Executor pilla el agente y las tools y usa el agente para decidir que tool usar y en que orden.
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                    tools=tools,
                                                    verbose=True,
                                                    memory=memory)

agent_executor.run('¿Cuales son los sintomas de la epilepsia?')
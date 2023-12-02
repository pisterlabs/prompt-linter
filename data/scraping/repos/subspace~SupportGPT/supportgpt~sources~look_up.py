from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.output_parsers import GuardrailsOutputParser
from langchain.prompts import PromptTemplate
from rich import print
import openai


#can add format function if needed 
rail_spec = """
<rail version="0.1">
    <output>
        <string name ='text' description="generated text" />
    </output>
    <prompt>
    Given the following user question, choose the best knowledge base from the list below
    that would contain the information to best answer the user's question. 
    You can only choose from the 3 options below in the list (research, forums, docs). 
    Only return the chosen option.

    List of knowledge bases and best use cases: 
    1. research - best used for questions about the future of the blockchain, but not questions about farming and bugs
    2. forums - best used for software bugs, wallet questions, and if there are pictures involved 
    3. docs - best used for standard farming and terminal command questions
    Let's work this out in a step by step way to be sure we have the right answer. 

    {{user_question}}

    
    </prompt>
</rail>

"""


user_question = """
I followed the steps and everything works very easily,
from gemini D I see the telemetry and the nodes are fine, 
but why have I never received anything in the accounts? 
Currently there are 8 nodes
"""

output_parser = GuardrailsOutputParser.from_rail_string(rail_spec)

print(output_parser.guard.base_prompt)

prompt = PromptTemplate(
    template=output_parser.guard.base_prompt,
    input_variables=output_parser.guard.prompt.variable_names,
)

# prompt to choose b/t different knowledge bases
llm=OpenAI(temperature=0)

output = llm(prompt.format_prompt(user_question=user_question).to_string())


print(output)

# agent_chain = initialize_agent(llm, 
# agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
# verbose=True)



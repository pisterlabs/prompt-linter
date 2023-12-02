from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Here is a brief description of {agent_name}:
{agent_description}

On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and \
10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy \
of the following {type_} for {agent_name}.

{type_}: {description}
Rating:"""

_prompt = PromptTemplate(input_variables=["agent_name", 
                                          "agent_description", 
                                          "description", 
                                          "type_"],
                         template=_template)

poignance_chain = LLMChain(prompt=_prompt, llm=llm)

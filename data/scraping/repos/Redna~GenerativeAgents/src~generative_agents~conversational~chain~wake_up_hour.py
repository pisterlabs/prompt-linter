import re
from langchain import LLMChain, PromptTemplate

from generative_agents.conversational.llm import llm
from generative_agents.conversational.output_parser.fuzzy_parser import FuzzyOutputParser, PatternWithDefault

_template = """
{agent_identity}

In general, {agent_lifestyle}
{agent_name}'s wake up hour:"""

_prompt = PromptTemplate(input_variables=["agent_name", 
                                          "agent_lifestyle",
                                          "agent_identity"],
                         template=_template)

_outputs = {"wake_up_hour": PatternWithDefault(pattern=re.compile(r"(\d{1,2}:\d{2} [ap]m)", re.IGNORECASE), 
                                              parsing_function=lambda x: x.upper(),
                                              default="6:00 AM")}

_output_parser = FuzzyOutputParser(output_definitions=_outputs)

wake_up_hour_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={"max_length": 30,
                                                                   "do_sample": True,
                                                                   "top_p": 0.95,
                                                                   "top_k": 60,
                                                                   "temperature": 0.4}
                                                                   , output_parser=_output_parser)

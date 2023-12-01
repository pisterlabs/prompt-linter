from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
[Conversation]
{conversation}

Write down if there is anything from the conversation that {agent} need to remember for her planning, from {agent} perspective, in a full sentence.

"{agent}"""

_prompt = PromptTemplate(input_variables=["agent",
                                          "conversation"],
                         template=_template)

planning_on_conversation_chain = LLMChain(prompt=_prompt, llm=llm)

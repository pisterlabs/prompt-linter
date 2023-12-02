from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Task: Turn the input into (subject, predicate, object).

Input: Sam Johnson is eating breakfast. 
Output: (Dolores Murphy, eat, breakfast) 
--- 
Input: Joon Park is brewing coffee.
Output: (Joon Park, brew, coffee)
---
Input: Jane Cook is sleeping. 
Output: (Jane Cook, is, sleep)
---
Input: Michael Bernstein is writing email on a computer. 
Output: (Michael Bernstein, write, email)
---
Input: Percy Liang is teaching students in a classroom. 
Output: (Percy Liang, teach, students)
---
Input: Merrie Morris is running on a treadmill. 
Output: (Merrie Morris, run, treadmill)
---
Input: {name} is {action_description}.
Output: ({name},"""


_prompt = PromptTemplate(input_variables=["name",
                                          "action_description"],
                            template=_template)

action_event_triple_chain = LLMChain(prompt=_prompt, llm=llm)

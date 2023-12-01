import re
from pydantic import BaseModel
from generative_agents import global_state
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


class ActionEventTriple(BaseModel):
    name: str
    address: str = None
    action_description: str

    async def run(self): 
      _prompt = PromptTemplate(input_variables=["name",
                                                "action_description"],
                                    template=_template)

      _action_event_triple_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
                                                        "max_new_tokens":20,
                                                        "do_sample": True,
                                                        "top_p": 0.95,
                                                        "top_k": 10,
                                                        "temperature": 0.4,
                                                        "cache_key": f"3action_event_triple_chain_{self.name}_{self.action_description}_{global_state.tick}"},
                                                        verbose=True)

      completion = await _action_event_triple_chain.arun(name=self.name, action_description=self.action_description)
      
      pattern = rf"Output: \(({self.name})\,(.*)\,(.*)\)"
      match = re.search(pattern, completion)
      
      if match:
        subject , predicate, object_ = match.groups()
        return (self.address if self.address else subject.strip(), predicate.strip(), object_.strip())
      
      return None

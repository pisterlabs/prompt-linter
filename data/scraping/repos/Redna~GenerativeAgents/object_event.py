import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Task: We want to understand the state of an object that is being used by someone.

Let's think step by step.
We want to know about {object_name}'s state.
Step 1. {name} is at/using the {object_name} {action_description}.
Step 2. Describe the {object_name}'s state: {object_name} is"""


class ObjectActionDescription(BaseModel):
    name: str
    object_name: str
    action_description: str

    async def run(self):

        _prompt = PromptTemplate(input_variables=["object_name",
                                                  "name",
                                                  "action_description"],
                                 template=_template)

        _object_event_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
            "max_new_tokens": 3,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 10,
            "temperature": 0.1,
            "cache_key": f"4_object_event_description_{self.name}_{self.action_description}_{global_state.tick}"}, verbose=True)

        completion = await _object_event_chain.arun(name=self.name, object_name=self.object_name, action_description=self.action_description)

        pattern = rf"({self.object_name} is .*)"
        return re.findall(pattern, completion)[-1]

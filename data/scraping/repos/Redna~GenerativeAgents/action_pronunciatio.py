import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Convert an action description to an emoji (important: use two or less emojis).

Action description: {action_description}
Emoji:"""


class ActionPronunciatio(BaseModel):
    action_description: str

    async def run(self):

        _prompt = PromptTemplate(input_variables=["action_description"],
                                 template=_template)

        _action_pronunciatio_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
            "max_new_tokens": 4,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 10,
            "temperature": 0.4,
            "cache_key": f"3action_pronunciatio_{self.action_description}_{global_state.tick}"}, verbose=True)

        completion = await _action_pronunciatio_chain.arun(action_description=self.action_description)

        pattern = rf"Emoji: (.*)"
        return re.findall(pattern, completion)[-1]

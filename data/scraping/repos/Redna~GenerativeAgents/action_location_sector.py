import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Choose an appropriate area  from the area options for a task at hand.

Sam Kim lives in [Sam Kim's house] that has Sam Kim's room, bathroom, kitchen.
Sam Kim is currently in [Sam Kim's house] that has Sam Kim's room, bathroom, kitchen. 
Area options: [Sam Kim's house, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy].
* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the "Area options," verbatim.
For taking a walk, Sam Kim should go to the following area: [Johnson Park]
---
Jane Anderson lives in [Oak Hill College Student Dormatory] that has Jane Anderson's room.
Jane Anderson is currently in [Oak Hill College] that has a classroom, library
Area options: [Oak Hill College Student Dormatory, The Rose and Crown Pub, Hobbs Cafe, Oak Hill College, Johnson Park, Harvey Oak Supply Store, The Willows Market and Pharmacy]. 
* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the "Area options," verbatim.
For eating dinner, Jane Anderson should go to the following area: [Hobbs Cafe]
--- 
{agent_name} lives in [{agent_home}] that has {agent_home_arenas}.
{agent_name} is currently in [{agent_current_sector}] that has {agent_current_sector_arenas}.
Area options: [{available_sectors_nearby}].
* Stay in the current area if the activity can be done there. Only go out if the activity needs to take place in another place.
* Must be one of the "Area options," verbatim.
For {curr_action_description}, {agent_name} should go to the following area: ["""


class ActionSectorLocations(BaseModel):
    agent_name: str
    agent_home: str
    agent_home_arenas: str
    agent_current_sector: str
    agent_current_sector_arenas: str
    available_sectors_nearby: str
    curr_action_description: str

    async def run(self):
        _prompt = PromptTemplate(input_variables=["agent_name",
                                                  "agent_home",
                                                  "agent_home_arenas",
                                                  "agent_current_sector",
                                                  "agent_current_sector_arenas",
                                                  "available_sectors_nearby",
                                                  "curr_action_description"],
                                 template=_template)

        _action_sector_locations_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
            "max_new_tokens": 10,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 60,
            "temperature": 0.4,
            "cache_key": f"2action_sector_locations_{self.agent_name}_{global_state.tick}"}, verbose=True)

        completion = await _action_sector_locations_chain.arun(agent_name=self.agent_name,
                                                               agent_home=self.agent_home,
                                                               agent_home_arenas=self.agent_home_arenas,
                                                               agent_current_sector=self.agent_current_sector,
                                                               agent_current_sector_arenas=self.agent_current_sector_arenas,
                                                               available_sectors_nearby=self.available_sectors_nearby,
                                                               curr_action_description=self.curr_action_description)

        pattern = rf"{self.agent_name} should go to the following area: \[(.*)\]"
        return re.findall(pattern, completion)[-1]

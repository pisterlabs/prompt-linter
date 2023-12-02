import re
from typing import List, Tuple

from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

hours = ["00:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", 
         "05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", 
         "10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM", 
         "03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM",
         "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]

_template = """
Hourly schedule format:
{hourly_schedule_format}

{identity}

Here the originally intended hourly breakdown 
{name}'s schedule for today:
{intended_schedule}

{prior_schedule}
{current_activity}"""

class HourlyBreakdown(BaseModel):
    
    identity: str
    current_hour: str
    wake_up_hour: str
    name: str
    today: str
    hourly_organized_activities: List[str]
    actual_activities: List[str]

    def _build_hourly_schedule_format(self) -> str:
        formatted_hours = [f"{self.today} -- {hour} | Activity: [Fill in]" for hour in hours]
        return "\n".join(formatted_hours)

    def _build_intended_schedule(self) -> str: 
        statements = [f"{i+1}) {hourly_organized_activity}" for i, hourly_organized_activity in enumerate(self.hourly_organized_activities)]
        return ", ".join(statements)

    def _build_prior_schedule(self, activities) -> str: 
        formatted_hours = [f"{self.today} -- {hour} | Activity: {activity[0]}" for hour, activity in zip(hours, activities)]
        return "\n".join(formatted_hours)

    async def run(self): 
        wake_up_index = hours.index(self.wake_up_hour.zfill(8))
        hourly_schedule_format = self._build_hourly_schedule_format()

        # TODO think about asyncio tasks and make 3 plans in parallel - Use the llm to rate the best fitting plan
        activities_hourly_organized = []

        _prompt = PromptTemplate(input_variables=["hourly_schedule_format", "identity", "name", "intended_schedule", "prior_schedule", "current_activity"],
                                template=_template)
        _hourly_breakdown_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
                                                "max_new_tokens":50,
                                                "do_sample": True,
                                                "top_p": 0.95,
                                                "top_k": 60,
                                                "temperature": 0.4}
                                                , verbose=True)
        
        for i, hour in enumerate(hours):
            
            if i < wake_up_index:
                activities_hourly_organized += [("sleeping", 60)]
            else:
                intended_schedule = self._build_intended_schedule()
                prior_schedule = self._build_prior_schedule(activities_hourly_organized)
                current_activity = f"{self.today} -- {hour} | Activity: {self.name} is"


                _hourly_breakdown_chain.llm_kwargs["cache_key"] = f"hourly_breakdown_{self.name}_{hour}_{global_state.tick}"

                completion = await _hourly_breakdown_chain.arun(hourly_schedule_format=hourly_schedule_format, 
                                                                identity=self.identity, 
                                                                name=self.name, 
                                                                intended_schedule=intended_schedule, 
                                                                prior_schedule=prior_schedule, 
                                                                current_activity=current_activity)

                pattern = rf"{hour} \| Activity: {self.name} is (.*)"
                activities_hourly_organized += [(re.findall(pattern, completion)[-1], 60)]

        return activities_hourly_organized



import re
from pydantic import BaseModel
from generative_agents import global_state
from generative_agents.conversational.llm import llm
from langchain import LLMChain, PromptTemplate

_template = """
Current activity: sleep in bed
Objects available: [bed, easel, closet, painting]
From the objects available above, pick ONE most relevant object: bed
---
Current activity: painting
Objects available: [easel, closet, sink, microwave]
From the objects available above, pick ONE most relevant object: easel
---
Current activity: cooking
Objects available: [stove, sink, fridge, counter]
From the objects available above, pick ONE most relevant object: stove
---
Current activity: watch TV
Objects available: [couch, TV, remote, coffee table]
From the objects available above, pick ONE most relevant object: TV
---
Current activity: study
Objects available: [desk, computer, chair, bookshelf]
From the objects available above, pick ONE most relevant object: desk
---
Current activity: talk on the phone
Objects available: [phone, charger, bed, nightstand]
From the objects available above, pick ONE most relevant object: phone
---
Current activity: {action_description}
Objects available: [{available_objects}]
From the objects available above, pick ONE most relevant object:"""

class ActionLocationGameObject(BaseModel):
    action_description: str
    available_objects: str
    retry: str = 0

    async def run(self): 
      _prompt = PromptTemplate(input_variables=["action_description",
                                          "available_objects"],
                            template=_template)

      _action_location_game_object_chain = LLMChain(prompt=_prompt, llm=llm, llm_kwargs={
                                                      "max_new_tokens": 3, 
                                                      "do_sample": True,
                                                      "num_beams": 1,
                                                      "cache_key": f"2action_location_game_object_{self.retry}_{self.action_description}_{global_state.tick}"},
                                                      verbose=True)

      completion = await _action_location_game_object_chain.arun(action_description=self.action_description,
                                                                 available_objects=self.available_objects)
      
      pattern = rf"pick ONE most relevant object: (.*)"
      object_ = re.findall(pattern, completion)[-1]
      print(">>>>>>>> " + object_)
      return object_




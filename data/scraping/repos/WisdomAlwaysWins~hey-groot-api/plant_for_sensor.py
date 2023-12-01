import json
import random
import pickle
import numpy as np
from langchain.tools import BaseTool

JSON_PATH = 'static/sensor_response_data.json'
MODEL_PATH = 'static/sentence_transformer_model.pkl'

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

class PlantSensor(BaseTool):
    name = "Plant_For_Sensor"
    description = """조건에 맞는 응답을 가져올 때 사용하는 도구이다. 이때, 배열에 있는 응답 중 하나를 랜덤으로 추출하고 그대로 응답한다."""  
    data : list
    plant_type : str
    
    def __init__(self, data : list = [0,0,0,0], plant_type= "") :
      super(PlantSensor, self).__init__(data = data, plant_type = plant_type)
      # print(plant_type, data)
    
    def _run(self, query: str) -> str:
        return get_response(query, arduino_data=self.data, plant_type=self.plant_type)
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("질문에 답할 수 없어요.")


def get_response(query, arduino_data, plant_type):
  
    def get_most_similar_question(query, plant_type):
      input = plant_type + " " + query
      input_embedding = model.encode(input)
      similarities = {}
      for plant in data['plants'].keys():
          plant_embedding = model.encode(plant)
          similarity = np.dot(input_embedding, plant_embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(plant_embedding))
          similarities[plant] = similarity
      most_similar_plant = max(similarities, key=similarities.get)
      return most_similar_plant
        
    current_conditions = {
      "temperature": arduino_data[0], 
      "humidity": arduino_data[1], 
      "illumination": arduino_data[2], 
      "moisture": arduino_data[3]
      }
    
    responses = data["plants"].get(get_most_similar_question(query, plant_type), {}).get("environment_responses", [])
    
    answers = data["responses"]
    
    valid_responses = []
    
    for response in responses:
        conditions = response["conditions"]
        
        for condition in conditions :
          
          if (conditions[condition]["low"] <= current_conditions[condition] <= conditions[condition]["high"]) :
            if condition == "temperature" : valid_responses.extend(answers["mid"]["temperature"].values())
            elif condition == "humidity" : valid_responses.extend(answers["mid"]["humidity"].values())
            elif condition == "illumination" : valid_responses.extend(answers["mid"]["illumination"].values())
            elif condition == "moisture" : valid_responses.extend(answers["mid"]["moisture"].values())
              
          elif (current_conditions[condition] < conditions[condition]["low"]) :
            if condition == "temperature" : valid_responses.extend(answers["low"]["temperature"].values())
            elif condition == "humidity" : valid_responses.extend(answers["low"]["humidity"].values())
            elif condition == "illumination" : valid_responses.extend(answers["low"]["illumination"].values())
            elif condition == "moisture" : valid_responses.extend(answers["low"]["moisture"].values())
            
          elif (conditions[condition]["high"] < current_conditions[condition]) :
            if condition == "temperature" : valid_responses.extend(answers["high"]["temperature"].values())
            elif condition == "humidity" : valid_responses.extend(answers["high"]["humidity"].values())
            elif condition == "illumination" : valid_responses.extend(answers["high"]["illumination"].values())
            elif condition == "moisture" : valid_responses.extend(answers["high"]["moisture"].values())
        
        '''        
        mid_is_valid = all(
            conditions[condition]["low"] <= current_conditions[condition] <= conditions[condition]["high"]
            for condition in conditions
        )
        
        low_is_valid = all(
          current_conditions[condition] < conditions[condition]["low"] 
          for condition in conditions
        )
        
        high_is_valid = all(
          conditions[condition]["high"] < current_conditions[condition] 
          for condition in conditions
        )
      '''

    all_responses = []
    for resp in valid_responses:
        for condition, value in current_conditions.items():
            # print(condition, value)
            resp = resp.replace(f"${{{condition}}}", str(value))
        all_responses.append(resp)
    
    if all_responses:
        return all_responses
    
    return ["지금은 응답을 해드릴 수 없어요."]




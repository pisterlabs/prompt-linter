import requests
import json
from langchain.agents import Tool

def ontology(input):
  # input is ignored
  response = requests.get('https://dev.blawx.com/jason/privacy-act/test/permitted_use/onto/')
  #print(response)
  package = json.loads(response.text)
  output = "The categories which take only an object as a parameters are " + ", ".join(package['Categories']) + ".\n"
  output = "The attributes that take only an object are " + ", ".join([(a['Attribute'] + " which applies to an object of category " + a['Category']) for a in package['Attributes'] if a['Type'] == "boolean"]) + ".\n"
  output += "The attributes that take an object and a value are " + ', '.join([(a['Attribute'] + " which applies to an object of category " + a['Category'] + " and accepts a value of type " + a['Type']) for a in package['Attributes'] if a['Type'] != "boolean"]) + '.\n'
  output += "The relationships I know about are "
  for r in package['Relationships']:
    output += r['Relationship'] + ", which accepts "
    index = 1
    while "Parameter"+str(index) in r:
      output += "a " + r['Parameter'+str(index)] + ","
      index += 1
    output += ";"
  return output

ontology_tool = Tool.from_function(
        func=ontology,
        name="Ontology",
        description="useful for when you need to know the categories, attributes, and relationships available in the permitted_uses tool. Does not require input."
    )
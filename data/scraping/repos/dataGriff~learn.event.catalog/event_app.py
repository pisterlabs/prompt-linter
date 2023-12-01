import json 
import openai
import os

openai.api_key = os.getenv("CHATGPT_KEY")

architecture_required = "dog walking"
architecture_required_path = architecture_required.replace(" ","_")
version = "1"
event = "DogProfileCreated"

template_md = """
---
name: AddedItemToCart
version: 0.0.2
summary: |
  Holds information about what the user added to their shopping cart.
producers:
    - Basket Service
consumers:
    - Data Lake
owners:
    - dboyne
    - mSmith
---

<Admonition>When firing this event make sure you set the `correlation-id` in the headers. Our schemas have standard metadata make sure you read and follow it.</Admonition>

### Details

This event can be triggered multiple times per customer. Everytime the customer adds an item to their shopping cart this event will be triggered.

We have a frontend application that allows users to buy things from our store. This front end interacts directly with the `Basket Service` to add items to the cart. The `Basket Service` will raise the events.

<NodeGraph title="Consumer / Producer Diagram" />

<EventExamples title="How to trigger event" />

<Schema />

<SchemaViewer renderRootTreeLines defaultExpandedDepth='0' maxHeight="500" />
"""

template_json = """
{
  "$id": "https://example.com/AddedItemToCart.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AddedItemToCart",
  "type": "object",
  "properties": {
    "metadata": {
      "type": "object",
      "properties": {
        "correlationId": {
          "type": "string",
          "description": "The ID of the user"
        },
        "domain": {
          "type": "string",
          "description": "The domain of the event"
        },
        "service": {
          "type": "string",
          "description": "The name of the service that triggered the event"
        }
      },
      "required": ["correlationId", "domain"]
    },
    "data": {
      "type": "object",
      "properties": {
        "userId": {
          "type": "string",
          "description": "The ID of the user"
        },
        "itemId": {
          "type": "string",
          "description": "The ID of the shopping item"
        },
        "quantity": {
          "type": "number",
          "description": "How many items the user wants to add to their shopping cart",
          "minimum": 1,
          "maximum": 1000,
          "default": 1
        }
      }
    }
  }
}
"""

file=open(f"event_lists/{architecture_required_path}/{architecture_required_path}_v{version}.json","r")

text = file.read()

event_list = json.loads(text)

unique_teams = set(d['team'] for d in event_list)

for event in event_list:
    domain =  event["domain"]
    print(domain)
    event_name =  event["event"]
    print(event_name)
    domain_path = domain.replace(" ","_")
    chat_completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": f"Return the markdown only in your response for the event {event_name} in the context of {architecture_required} using this markdown as the example template {template_md} with owner values being the appropriate values only from this list {unique_teams}"}])
    
    out = chat_completion['choices'][0]['message']['content']

    event_path = event_name.replace(" ","_")
    full_path = f"domains/{architecture_required_path}_v{version}/{domain_path}/events/{event_name}"
    
    isExist = os.path.exists(full_path)
    if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(full_path)
        print("The new directory is created!")
        

    with open(f"{full_path}/index.md", "w") as outfile:
        outfile.write(out)
        
    chat_completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": f"Return the json schema only in your response for the event {event_name} in the context of {architecture_required} using this json as the example template {template_json}"}])
    
    out = chat_completion['choices'][0]['message']['content']

    with open(f"{full_path}/schema.json", "w") as outfile:
        outfile.write(out)
        
        
        
        


from openai import OpenAI
import os
import json
import time

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

#client=OpenAI()
#client.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def getCurrentWeather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


assistant = client.beta.assistants.create(
  instructions="You are a weather bot. Use the provided functions to answer questions.",
  model="gpt-4-1106-preview",
  tools=[{
    "type": "function",
    "function": {
      "name": "getCurrentWeather",
      "description": "Get the weather in location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "The city and state e.g. San Francisco, CA"},
          "unit": {"type": "string", "enum": ["c", "f"]}
        },
        "required": ["location"]
      }
    }
  }, {
    "type": "function",
    "function": {
      "name": "getNickname",
      "description": "Get the nickname of a city",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "The city and state e.g. San Francisco, CA"},
        },
        "required": ["location"]
      }
    } 
  }]
)

user_prompt="洛杉矶的温度多少?"


thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=user_prompt
)


run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions="You are a weather bot. Use the provided functions to answer questions.",
)

print(message)

while True:
    run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id)
    if run.status == "requires_action":
        break

#print(run)  
#time.sleep(30)   
# Assuming 'run' is already retrieved and has the structure as mentioned

# Check if 'required_action' and 'submit_tool_outputs' exist in the 'run' object
if hasattr(run, 'required_action') and hasattr(run.required_action, 'submit_tool_outputs'):
    tool_calls = run.required_action.submit_tool_outputs.tool_calls

    # Iterate through each tool call and extract the required information
    for tool_call in tool_calls:
        tool_call_id = tool_call.id
        tool_type = tool_call.type
        function_name = tool_call.function.name
        function_arguments = tool_call.function.arguments

        # Print the extracted information
        print(f"Tool Call ID: {tool_call_id}")
        print(f"Tool Type: {tool_type}")
        print(f"Function Name: {function_name}")
        print(f"Function Arguments: {function_arguments}")
else:
    print("Required action or tool calls not found in the run object.")


import json

arguments_dict = json.loads(function_arguments)

# Get the function object from the current global scope
function_to_call = globals().get(function_name)

# Check if the function exists
if function_to_call:
    # Call the function with unpacked arguments
    result = function_to_call(**arguments_dict)
    print(result)
else:
    print(f"Function {function_name} not found.")
    
data = json.loads(result)

# Extract the value associated with the key 'temperature'
temperature = data.get("temperature")
print(temperature)
# Check if the temperature key exists and print the value
'''if temperature is not None:
    print(f"The temperature is {temperature} degrees.")
else:
    print("Temperature key not found in the data.")'''

run = client.beta.threads.runs.submit_tool_outputs(
    thread_id=thread.id,
    run_id=run.id,
    tool_outputs=[
      {
        "tool_call_id": tool_call_id,
        "output": temperature,
      },
      
    ]
    ) 
   
while True:
    run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id)
    if run.status == "completed":
        break

print(run.status)
messages = client.beta.threads.messages.list(thread_id=thread.id)
first_message_value = messages.data[0].content[0].text.value
#first_message_value = messages.data[0].content[0]
print(first_message_value)
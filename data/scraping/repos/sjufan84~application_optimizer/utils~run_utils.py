""" Utilities to support the run endpoints """
import os
import time
import json
#from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
api_key = os.getenv("OPENAI_KEY2")
organization = os.getenv("OPENAI_ORG2")

# Create the OpenAI client
client = OpenAI(api_key=api_key, organization=organization, max_retries=3, timeout=100)

available_functions = {
    "functions" : {
    #"get_trends_data": get_trends_data
    }
}

def call_named_function(function_name: str, **kwargs):
    try:
        # Check if the function name exists in the dictionary
        if function_name in available_functions["functions"]:
            # Call the function with unpacked keyword arguments
            return available_functions["functions"][function_name](**kwargs)
        else:
            return f"Function {function_name} not found."
    except TypeError as e:
        return f"Error in calling {function_name}: {e}"
'''

class Message(BaseModel):
    role: str = "user"
    content: str = Field(..., description="The content of the message")

# Create the endpoint url for the function we want to call
fastapi_base_url = "http://localhost:8000"

# Define a function to create a run
def create_run(thread_id: str, assistant_id: str):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )
    return run
'''

def poll_run_status(run_id: str, thread_id: str):
    run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    tool_return_values = []

    while run_status.status not in ["completed", "failed", "expired", "cancelling", "cancelled"]:
        if run_status.status == "requires_action":
            # Handle the action required by the assistant
            tool_outputs = []
            
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls

            for tool_call in tool_calls:
                # Extract necessary details
                function_name = tool_call.function.name
                tool_call_id = tool_call.id
                parameters = json.loads(tool_call.function.arguments)

                # Call the function
                function_output = call_named_function(function_name=function_name, **parameters)
                
                # Append the tool output
                tool_outputs.append({
                    "tool_call_id": tool_call_id,
                    "output": function_output
                })
                tool_return_values.append({
                    "tool_name" : function_name,
                    "tool_call_id": tool_call_id,
                    "output": function_output
                })

            # Submit the tool outputs to the run
            run = client.beta.threads.runs.submit_tool_outputs(thread_id=thread_id, run_id=run_id, tool_outputs=tool_outputs)
            run_status = run
        else:
            # If the status is "queued" or "in-progress", wait and then retrieve status again
            time.sleep(1)
            run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

    # Gather the final messages after completion
    final_messages = client.beta.threads.messages.list(thread_id=thread_id, limit=1)

    return {
        "thread_id": thread_id, 
        "message": final_messages.data[0].content[0].text.value,
    }

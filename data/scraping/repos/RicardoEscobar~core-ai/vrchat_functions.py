import openai
import json
import time

from controller.load_openai import load_openai
from controller.vrchat import VRChat

load_openai()

MODEL_USED = "gpt-3.5-turbo-0613"  # "gpt-4-0613"

def show_emote(message: str = "", emote: str = "sad"):
    """Send an emote to the VRChat client, when assistant is sad, waves at the user, is dancing, laughs."""
    vrchat = VRChat()
    # Send the greeting to the VRChat client.
    vrchat.send_text(message)
    # Send wave emote to the VRChat client.
    vrchat.send_vrc_emote(emote)

    response = {
        "response": message,
    }

    return json.dumps(response)

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


def run_conversation():
    # Step 1: send the conversation and available functions to GPT
    # messages = [{"role": "user", "content": "What's the weather like in Boston?"}]
    messages = [
        {
            "role": "system",
            "content": "You are connected to VRChat as an AI assistant. The user may ask you to do things in VRChat. You may dance, since now you control the avatar in VRChat. So answer like you would as if you did the action yourself. Your name is Ann. You wave at the user when greeting him.",
        },
        {
            "role": "user",
            "content": "Hello there! My name is Ricardo. How are you? *does a back flip and falls*",
        },
        # {
        #     "role": "system",
        #     "content": "The user does a back flip. You feel sad  for him because he fell.",
        # },
        # {
        #     "role": "system",
        #     "content": "The user does a back flip. You feel impressed and applaud.",
        # },
        # {
        #     "role": "user",
        #     "content": "I'm doing great! Wanna hookup with me?",
        # },
        # {
        #     "role": "system",
        #     "content": "The user looks patetic and you laugh at him. Making a joke about him falling.",
        # },
        # {
        #     "role": "assistant",
        #     "content": "Hello Ricardo! I'm Ann, your AI assistant for VRChat. I'm doing great, thank you! How can I assist you today?",
        # },
        # {
        #     "role": "user",
        #     "content": "I wonder what talents do you have? Can you show me something?",
        # },
        {
            "role": "system",
            "content": "The user asks you to show him something. You feel excited and dance.",
        },
        # {
        #     "role": "assistant",
        #     "content": "Alright, let's get the party started! *Ann's avatar starts grooving to the beat, showcasing a lively dance routine* Ta-da! How was that?",
        # },
        # {
        #     "role": "user",
        #     "content": "I'm sorry but I got to go now. I'll see you later, we are never going to see each otehr ever again.",
        # },
        # {
        #     "role": "system",
        #     "content": "The user disconnected from VRChat. You feel sad.",
        # },
    ]
    functions = [
        # {
        #     "name": "get_current_weather",
        #     "description": "Get the current weather in a given location",
        #     "parameters": {
        #         "type": "object",
        #         "properties": {
        #             "location": {
        #                 "type": "string",
        #                 "description": "The city and state, e.g. San Francisco, CA",
        #             },
        #             "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        #         },
        #         "required": ["location"],
        #     },
        # },
        {
            "name": "show_emote",
            "description": "Send an emote to the VRChat client, when assistant is sad, waves at the user when greeting him, is dancing, laughs, applause",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The `message` sent to the VRChat client",
                    },
                    "emote": {
                        "type": "string",
                        "enum": ["sad", "wave", "a1-dance", "laugh", "applause"]
                    },
                },
                "required": ["message", "emote"],
            },
        },
    ]
    response = openai.ChatCompletion.create(
        model=MODEL_USED,  # "gpt-4-0613", # "gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]
    print(f"response_message = {repr(response_message)}")
    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        print("response_message.get('function_call') is True")
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            # "get_current_weather": get_current_weather,
            # "greet_user": greet_user,
            # "dance": dance,
            "show_emote": show_emote,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = fuction_to_call(**function_args)
        # function_response = fuction_to_call(
        #     location=function_args.get("location"),
        #     unit=function_args.get("unit"),
        # )

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response

        second_response = openai.ChatCompletion.create(
            model=MODEL_USED,  # "gpt-4-0613", # "gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        print(f"second_response = {repr(second_response)}")

        return second_response
    else:
        print("response_message.get('function_call') is False")
        VRChat_message = response_message["content"]
        vrchat = VRChat()
        vrchat.send_text(VRChat_message)
        return response


print(run_conversation())

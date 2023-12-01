import json
from typing import Dict, List, Tuple
import re

from chatbot.system_messages.system import (
    function_res_agent, 
    base_system_message,
    )
from chatbot.agents.function_mapper import FunctionMapper
from chatbot.agents.function_call_agent import FunctionCallAgent
from chatbot.agents.base_agent import RegularAgent
from chatbot.agents.function_response_agent import function_response_agent
from chatbot.agents.anthropic_base import AnthropicAgent

function_mapper = FunctionMapper()
fc = FunctionCallAgent()
ra = RegularAgent()
aa = AnthropicAgent()
function_call_agent = fc.call
regular_agent = ra.call
anthropic_agent = aa.call

def get_command(prompt):
    """Extract command from prompt if it exists"""
    match = re.search(r"/\w+", prompt)
    if match:
        return match.group(0)
    return None

def call_function(function_name, function_args):
    if function_name not in function_mapper.function_map:
        raise Exception(f"Function {function_name} not found.")
    if function_args is None:
        return function_mapper.function_map[function_name]()
    return function_mapper.function_map[function_name](*function_args.values())

def run_conversation(prompt: str, conversation: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]], int, float]:
    agents = function_mapper.agents
    command = get_command(prompt)
    if command in agents:
        agent_properties = agents[command]
        prompt = prompt[agent_properties["command_length"]:].strip()
        if agent_properties["is_function"]:
            print(agent_properties["name"])
            response = function_call_agent(
                prompt=prompt,
                conversation=conversation,
                system_message=agent_properties["agent"].system_message,
                function_params=agent_properties["agent"].function_params,
                temperature=agent_properties["agent"].temperature,
                top_p=agent_properties["agent"].top_p,
                frequency_penalty=agent_properties["agent"].frequency_penalty,
                model=agent_properties["agent"].model,
            )
            message = response[0]
        
        else:
            print(agent_properties["name"])
            response = regular_agent(
                prompt=prompt,
                conversation=conversation,
                system_message=agent_properties["system_message"],
                stream=True,
            )
            return response
        
    elif command == "/claude":
        print('Anthropic')
        prompt = prompt[len(command):].strip()
        response = anthropic_agent(
            prompt=prompt, 
            conversation=conversation,
            stream=True
            )
        return response
    else:
        print('Personal Assistant')
        response = regular_agent(
            prompt=prompt, 
            conversation=conversation, 
            system_message=base_system_message,
            stream=True
            )
        return response

    if message.get("function_call"):
        function_name = message["function_call"]["name"]
        function_args = json.loads(message["function_call"]["arguments"])
        print(f"Function name: {function_name}")
        print(f"Function arguments: {function_args}")
        
        function_response = call_function(function_name, function_args)
        if function_name in function_mapper.functions_that_append_to_conversation:
            print(f"Function response: {function_response}")
            conversation.append({
                "role": "assistant",
                "content": function_response,  # directly add function response to the conversation
            })
            return function_response
        else:
            print(f"Function response: {function_response}")
            second_response = function_response_agent(
                prompt=prompt, 
                system_message=function_res_agent, 
                function_name=function_name, 
                function_response=function_response, 
                message=message,
                stream=True
                )
            return second_response

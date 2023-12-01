import openai
import json
from concurrent.futures import ThreadPoolExecutor
import os
from termcolor import colored
import random
from enum import StrEnum
from dataclasses import dataclass
import time
class OAIModels(StrEnum):
    GPT4 = "gpt-4",
    GPT3_5_Turbo_16k = "gpt-3.5-turbo-16k"
    GPT3_5_Turbo = "gpt-3.5-turbo"

@dataclass
class SwarmConfig:
    num_agents: int = 3
    max_response_tokens: int = 7000
    final_response_length: int = 2000
    iterations: int = 2
    system_message_model: OAIModels = OAIModels.GPT4
    answer_generation_model: OAIModels = OAIModels.GPT3_5_Turbo
    answer_iteration_model: OAIModels = OAIModels.GPT3_5_Turbo_16k
    response_aggregation_model: OAIModels = OAIModels.GPT3_5_Turbo_16k
    temperature: float = 0.4


class OpenAIApiCaller:
    def __init__(self, config: SwarmConfig):
        self.config = config

    def call_api(
        self,
        model,
        messages,
        temperature,
        max_tokens,
        functions=None,
        function_call=None,
    ):
        gpt_call_parameters = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if functions:
            gpt_call_parameters["functions"] = functions

        if function_call:
            gpt_call_parameters["function_call"] = function_call

        try:
            response = openai.ChatCompletion.create(**gpt_call_parameters)
            responses = ""
            for chunk in response:
                if function_call:
                    if chunk["choices"][0]["delta"].get("function_call"):
                        chunk = chunk["choices"][0]["delta"]
                        arguments_chunk = chunk["function_call"]["arguments"]
                        print(arguments_chunk, end="", flush=True)
                        responses += arguments_chunk
                else:
                    response_content = (
                        chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                    )
                    if response_content:
                        responses += response_content
            return responses
        except Exception as e:
            print(f"Error during API call: {e}")
            print(
                f"Payload: {json.dumps(gpt_call_parameters, indent=4)}"
            )  # Print the payload for debugging.
            return None


class GPTSwarm:
    def __init__(
        self,
        user_message,
        config: SwarmConfig,
    ):
        self.config = config
        self.n_system_messages = config.num_agents
        self.user_message = user_message
        self.gpt_api_caller = OpenAIApiCaller(config)
        self.system_messages, self.name_of_the_project = self.generate_system_messages(config.system_message_model)
        # Check and create folders
        self.base_directory = "output"
        timestamp = time.strftime("%Y%m%d_%H%M%S")  
        self.project_directory = os.path.join(
            self.base_directory, f"{self.name_of_the_project}_{timestamp}"
        )
        self._ensure_directory_exists(self.base_directory)
        self._ensure_directory_exists(self.project_directory)
        self.iterations_data = {}

    def _ensure_directory_exists(self, path):
        """Utility method to create directory if it doesn't exist."""

        if not os.path.exists(path):
            os.makedirs(path)

    def generate_system_messages(self, system_model: str):
        messages = [
            {
                "role": "system",
                "content": f"A system message is used to guide a large language model to solve a user's query. Provide a list with {self.n_system_messages} distinct and unique system messages to guide a large language model to solve the user's problem from different perspectives. System message addresses the llm as 'you' and gives it a role, direction and scope. Each system message should be unique and provide a different perspective on the user's problem. Give the llm a distinct role('by refering to it as you') with a unique perspective each time.Take a deep breath.",
            },
            {"role": "user", "content": f"{self.user_message}"},
        ]
        function_def = [
            {
                "name": "generateSystemMessages",
                "description": f"Generates {self.n_system_messages} varied and distinct system messages to guide a large language model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "system_messages": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": f"{self.n_system_messages} Distinct and unique system messages",
                            },
                        },
                        "name_of_the_project": {
                            "type": "string",
                            "description": "briefly describe the name for the project and include a random number",
                        },
                    },
                    "required": ["system_messages", "name_of_the_project"],
                },
            }
        ]
        function_call = {"name": "generateSystemMessages"}
        responses = self.gpt_api_caller.call_api(
            system_model,
            messages,
            temperature=self.config.temperature,
            max_tokens=2000,
            functions=function_def,
            function_call=function_call,
        )
        if responses:
            system_messages, name_of_the_project = (
                json.loads(responses)["system_messages"],
                json.loads(responses)["name_of_the_project"],
            )
            return system_messages, name_of_the_project
            return function_return
        return []

    def gpt_call(self, system_message: str):
        word_limit = self.config.max_response_tokens - self.config.final_response_length - (self.config.max_response_tokens// self.config.num_agents)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": self.user_message},
            {
                "role": "user",
                "content": f"please answer using no more than {word_limit} words. Do your best given the word constraint.Take a deep breath and lets work through this step by step. Return only your answer and nothing else.",
            },
        ]
        return self.gpt_api_caller.call_api(
            self.config.answer_generation_model, messages, self.config.temperature, word_limit
        )

    def write_responses_to_json(self, all_responses):
        responses_dict = {k: v for k, v in zip(self.system_messages, all_responses)}
        output_path = os.path.join(self.project_directory, "all_responses.json")
        with open(output_path, "w") as outfile:
            json.dump(responses_dict, outfile, indent=4)

    def synthesize_responses(self, responses):
        joined_responses = "\n".join(responses)
        prompt = f"Based on the following responses, provide a comprehensive and concise synthesis:\n{joined_responses}"
        messages = [
            {
                "role": "system",
                "content": "You are tasked with synthesizing multiple answers into one comprehensive response. Read each one carefully and synthesize them. Take a deep breath and lets work through this step by step. Just return the synthesis and nothing else",
            },
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": f"original user message: \n\n {self.user_message}",
            },
        ]
        synthesized_output = self.gpt_api_caller.call_api(
            self.config.response_aggregation_model, messages, max_tokens=self.config.final_response_length, temperature=self.config.temperature
        )
        # write the synthesized output to file
        print(colored("\nWrote synthesized response to file", "magenta"))
        output_path = os.path.join(self.project_directory, "synthesized_response.txt")
        with open(output_path, "w") as file:
            file.write(synthesized_output.strip())
        return synthesized_output.strip()

    def feedback_loop(self, current_responses):
        print(colored("\nStarting feedback loop", "blue"))
        for iteration in range(self.config.iterations):
            with ThreadPoolExecutor() as executor:
                # Pair up each system_message with the entire current_responses list
                args = list(
                    zip(
                        self.system_messages,
                        [current_responses] * len(self.system_messages),
                    )
                )
                updated_responses = list(
                    executor.map(self.gpt_call_with_feedback, args)
                )
            current_responses = updated_responses
            # Save the updated responses for this iteration
            self.write_responses_to_iteration_file(updated_responses, iteration + 1)
        return current_responses

    def write_responses_to_iteration_file(self, responses, iteration):
        print(
            colored(
                f"\nWrote responses for iteration {iteration} out of {self.config.iterations} to file",
                "green",
            )
        )
        file_name = os.path.join(
            self.project_directory, f"all_responses_iteration_{iteration}.json"
        )
        responses_dict = {k: v for k, v in zip(self.system_messages, responses)}
        self.iterations_data[iteration] = responses_dict
        with open(file_name, "w") as outfile:
            json.dump(responses_dict, outfile, indent=4)
            

    def gpt_call_with_feedback(self, args):
        system_message, peer_responses = args
        print(colored("...Swarm Communicating...", "yellow"))
        random.shuffle(peer_responses)
        all_responses = "\n".join(peer_responses)
        prompt = f"Consider these other approaches carefully:\n{all_responses}\n\nrate them first and then.Write your solution to this problem having considered the best aspects of all other responses.Take a deep breath and lets work through this step by step."
        word_limit =self.config.max_response_tokens - self.config.final_response_length -  (self.config.max_response_tokens // self.config.num_agents)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": f"please answer using no more than {word_limit} words. Do your best given the word constraint. return only your answer and nothing else.",
            },
        ]
        return self.gpt_api_caller.call_api(self.config.answer_iteration_model, messages, max_tokens=word_limit, temperature=self.config.temperature)

    def generate_response(self):
        print(colored("\n...Generating initial responses...", "red"))
        with ThreadPoolExecutor() as executor:
            initial_responses = list(executor.map(self.gpt_call, self.system_messages))

        self.write_responses_to_json(initial_responses)
        feedback_responses = self.feedback_loop(initial_responses)

        synthesized_response = self.synthesize_responses(feedback_responses)
        return synthesized_response, self.iterations_data
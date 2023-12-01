import openai
import json
from concurrent.futures import ThreadPoolExecutor
import os
from termcolor import colored
import random

# Search all my videos and find code download links easily: www.echohive.live


class Config:
    HOW_MANY_SYSTEM_MESSAGES = 3
    # this token limit is not used for system message generation or synthesis generation
    RESPONSE_TOKEN_LENGTH = 1000
    FEEDBACK_ROUNDS = 2
    MODEL_FOR_SYSTEM_MESSAGES_GENERATION = "gpt-4"
    MODEL_FOR_ANSWER_GENERATION = "gpt-4"
    MODEL_FOR_ANSWER_ITERATION = "gpt-3.5-turbo-16k"
    MODEL_FOR_SYNTHESIZING_FINAL_RESPONSE = "gpt-4"
    API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR_KEY_HERE"


class OpenAIApiCaller:
    def __init__(self):
        openai.api_key = Config.API_KEY

    def call_api(self, model, messages, temperature=0.3, max_tokens=Config.RESPONSE_TOKEN_LENGTH, functions=None, function_call=None):
        gpt_call_parameters = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        if functions:
            gpt_call_parameters["functions"] = functions

        if function_call:
            gpt_call_parameters["function_call"] = function_call

        try:
            response = openai.ChatCompletion.create(**gpt_call_parameters)
            responses = ''
            for chunk in response:
                if function_call:
                    if chunk["choices"][0]["delta"].get("function_call"):
                        chunk = chunk["choices"][0]["delta"]
                        arguments_chunk = chunk["function_call"]["arguments"]
                        print(arguments_chunk, end="", flush=True)
                        responses += arguments_chunk
                else:
                    response_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                    if response_content:
                        responses += response_content
            return responses
        except Exception as e:
            print(f"Error during API call: {e}")
            print(f"Payload: {json.dumps(gpt_call_parameters, indent=4)}")  # Print the payload for debugging.
            return None

class GPTSwarm:
    def __init__(self, user_message, gpt_api_caller, n_system_messages=Config.HOW_MANY_SYSTEM_MESSAGES):
        self.n_system_messages = n_system_messages
        self.user_message = user_message
        self.gpt_api_caller = OpenAIApiCaller()
        self.system_messages, self.name_of_the_project = self.generate_system_messages()
        # Check and create folders
        self.base_directory = "swarm_works"
        self.project_directory = os.path.join(self.base_directory, self.name_of_the_project)
        self._ensure_directory_exists(self.base_directory)
        self._ensure_directory_exists(self.project_directory)

    def _ensure_directory_exists(self, path):
        """Utility method to create directory if it doesn't exist."""
        if not os.path.exists(path):
            os.makedirs(path)

    def generate_system_messages(self):
        messages = [
            {
                "role": "system",
                "content": f"A system message is used to guide a large language model to solve a user's query. Provide a list with {self.n_system_messages} distinct and unique system messages to guide a large language model to solve the user's problem from different perspectives. System message addresses the llm as 'you' and gives it a role, direction and scope. Each system message should be unique and provide a different perspective on the user's problem. Give the llm a distinct role('by refering to it as you') with a unique perspective each time."
            },
            {"role": "user", "content": f"{self.user_message}"}
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
                            "description": f"{self.n_system_messages} Distinct and unique system messages"
                        }
                    },
                    "name_of_the_project": {
                        "type": "string",
                        "description": "briefly describe the name for the project"
                    }
                },
                "required": ["system_messages", "name_of_the_project"]
            }
        }
    ]
        function_call = {"name": "generateSystemMessages"}
        responses = self.gpt_api_caller.call_api(Config.MODEL_FOR_SYSTEM_MESSAGES_GENERATION, messages, max_tokens=2000, functions=function_def, function_call=function_call)
        if responses:
            system_messages, name_of_the_project = json.loads(responses)["system_messages"], json.loads(responses)["name_of_the_project"]
            return system_messages, name_of_the_project
            return function_return
        return []

    def gpt_call(self, system_message):
        word_limit = Config.RESPONSE_TOKEN_LENGTH - (Config.RESPONSE_TOKEN_LENGTH // 3)
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': self.user_message},
            {"role": "user", "content":f"please answer using no more than {word_limit} words. Do your best given the word constraint. Return only your answer and nothing else."},
        ]
        return self.gpt_api_caller.call_api(Config.MODEL_FOR_ANSWER_GENERATION, messages)

    def write_responses_to_json(self, all_responses):
        responses_dict = {k: v for k, v in zip(self.system_messages, all_responses)}
        output_path = os.path.join(self.project_directory, 'all_responses.json')
        with open(output_path, 'w') as outfile:
            json.dump(responses_dict, outfile, indent=4)

    def synthesize_responses(self, responses):
        joined_responses = "\n".join(responses)
        prompt = f"Based on the following responses, provide a comprehensive and concise synthesis:\n{joined_responses}"
        messages = [
            {'role': 'system', 'content': 'You are tasked with synthesizing multiple answers into one comprehensive response. Read each one carefully and synthesize them. Just return the synthesis and nothing else'},
            {'role': 'user', 'content': prompt},
            {"role": "user", "content": f"original user message: \n\n {self.user_message}"},
        ]
        synthesized_output = self.gpt_api_caller.call_api(Config.MODEL_FOR_SYNTHESIZING_FINAL_RESPONSE, messages, max_tokens=2000)
        # write the synthesized output to file
        print(colored("\nWrote synthesized response to file", "magenta"))
        output_path = os.path.join(self.project_directory, 'synthesized_response.txt')
        with open(output_path, 'w') as file:
            file.write(synthesized_output.strip())
        return synthesized_output.strip()

    def feedback_loop(self, current_responses):
        print(colored("\nStarting feedback loop", "blue"))
        for iteration in range(Config.FEEDBACK_ROUNDS):
            with ThreadPoolExecutor() as executor:
                # Pair up each system_message with the entire current_responses list
                args = list(zip(self.system_messages, [current_responses] * len(self.system_messages)))
                updated_responses = list(executor.map(self.gpt_call_with_feedback, args))
            current_responses = updated_responses
            # Save the updated responses for this iteration
            self.write_responses_to_iteration_file(updated_responses, iteration+1)
        return current_responses

    def write_responses_to_iteration_file(self, responses, iteration):
        print(colored(f"\nWrote responses for iteration {iteration} out of {Config.FEEDBACK_ROUNDS} to file", "green"))
        file_name  = os.path.join(self.project_directory, f'all_responses_iteration_{iteration}.json')
        responses_dict = {k: v for k, v in zip(self.system_messages, responses)}
        with open(file_name, 'w') as outfile:
            json.dump(responses_dict, outfile, indent=4)




    def gpt_call_with_feedback(self, args):
        system_message, peer_responses = args
        print(colored("...Swarm Communicating...", "yellow"))
        random.shuffle(peer_responses)
        all_responses = "\n".join(peer_responses)
        prompt = f"Consider these other approaches carefully:\n{all_responses}\n\nrate them first and then.Write your solution to this problem having considered the best aspects of all other responses."
        word_limit = Config.RESPONSE_TOKEN_LENGTH - (Config.RESPONSE_TOKEN_LENGTH // 3)
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': prompt},
            {"role": "user", "content":f"please answer using no more than {word_limit} words. Do your best given the word constraint. return only your answer and nothing else."},
        ]
        return self.gpt_api_caller.call_api(Config.MODEL_FOR_ANSWER_ITERATION, messages)

    def generate_response(self):
        print(colored("\n...Generating initial responses...", "red"))
        with ThreadPoolExecutor() as executor:
            initial_responses = list(executor.map(self.gpt_call, self.system_messages))

        self.write_responses_to_json(initial_responses)
        feedback_responses = self.feedback_loop(initial_responses)

        synthesized_response = self.synthesize_responses(feedback_responses)
        return synthesized_response

def main():
    print(colored("...INITIALIZING SWARM...", "red"))
    # user_message = "Write the full code for a snake game with pygame. 3 snakes move automatically towards the food. there is a user snake which is also competing. keep track of the scores. snakes wrap around the screen. "
    user_message = "create an interesting and unique strategy game outline using pygame. Outline needs to include all aspects and features of the game so that  adeveloper can build the full game by only looking at the outline. the game needs to be no more than 300 lines of of code so It can not be too complex."
    gpt_api_caller = OpenAIApiCaller()
    swarm = GPTSwarm(user_message, gpt_api_caller)
    # exit()
    synthesized_response = swarm.generate_response()
    print(synthesized_response)

if __name__ == "__main__":
    main()

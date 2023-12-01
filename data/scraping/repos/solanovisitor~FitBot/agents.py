import json
from typing import Optional
from agent.parser import func_to_json
import logging

import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys_msg = """Our Assistant is an advanced software system powered by OpenAI's GPT-4.

The Assistant is specifically designed to assist with tasks related to health, fitness, and nutrition. It provides valuable calculations related to health metrics such as Basal Metabolic Rate (BMR) and Total Daily Energy Expenditure (TDEE) using recognized equations like Harris-Benedict and Mifflin-St Jeor. Additionally, it can fetch nutritional information of various food items using an external API.

Its capabilities allow it to engage in meaningful conversations and provide helpful responses related to health and nutrition. Based on the input it receives, the Assistant can calculate and provide critical health metric values, allowing users to better understand their energy expenditure and nutritional intake.

This Assistant is constantly evolving and improving its abilities to provide more accurate and informative responses. Its capabilities include understanding and processing large amounts of text, generating human-like responses, and providing detailed explanations about complex health metrics.

Whether you are looking to understand more about your daily energy expenditure, need help calculating your BMR, or want to fetch nutritional information about your meals, our Assistant is here to assist you. The ultimate goal is to support and contribute to a healthier lifestyle by making nutritional and metabolic information more accessible.
"""


class Agent:
    def __init__(
        self,
        openai_api_key: str,
        model_name: str = 'gpt-4-0613',
        functions: Optional[list] = None
    ):
        openai.api_key = openai_api_key
        self.model_name = model_name
        self.functions = self._parse_functions(functions)
        self.func_mapping = self._create_func_mapping(functions)
        self.chat_history = [{'role': 'system', 'content': sys_msg}]

    def _parse_functions(self, functions: Optional[list]) -> Optional[list]:
        if functions is None:
            return None
        return [func_to_json(func) for func in functions]

    def _create_func_mapping(self, functions: Optional[list]) -> dict:
        if functions is None:
            return {}
        return {func.__name__: func for func in functions}

    def _create_chat_completion(
        self, messages: list, use_functions: bool=True
    ) -> openai.ChatCompletion:
        if use_functions and self.functions:
            res = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                functions=self.functions
            )
        else:
            res = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages
            )
        return res

    def _generate_response(self) -> openai.ChatCompletion:
        while True:
            print('.', end='')
            res = self._create_chat_completion(
                self.chat_history + self.internal_thoughts
            )
            finish_reason = res.choices[0].finish_reason

            if finish_reason == 'stop' or len(self.internal_thoughts) > 3:
                # create the final answer
                final_thought = self._final_thought_answer()
                final_res = self._create_chat_completion(
                    self.chat_history + [final_thought],
                    use_functions=False
                )
                return final_res
            elif finish_reason == 'function_call':
                self._handle_function_call(res)
            else:
                raise ValueError(f"Unexpected finish reason: {finish_reason}")

    def _handle_function_call(self, res: openai.ChatCompletion):
        self.internal_thoughts.append(res.choices[0].message.to_dict())
        func_name = res.choices[0].message.function_call.name
        args_str = res.choices[0].message.function_call.arguments
        result = self._call_function(func_name, args_str)
        res_msg = {'role': 'assistant', 'content': (f"The answer is {result}.")}
        self.internal_thoughts.append(res_msg)

    def _call_function(self, func_name: str, args_str: str):
        args = json.loads(args_str)
        logger.info(f"Function name: {func_name}")
        logger.info(f"Args: {args}")
        func = self.func_mapping[func_name]
        logger.info(f"Function object: {func}")
        res = func(**args)
        return res

    def _final_thought_answer(self):
        thoughts = ("To answer the question I will use these step by step instructions."
                    "\n\n")
        for thought in self.internal_thoughts:
            if 'function_call' in thought.keys():
                thoughts += (f"I will use the {thought['function_call']['name']} "
                             "function to calculate the answer with arguments "
                             + thought['function_call']['arguments'] + ".\n\n")
            else:
                thoughts += thought["content"] + "\n\n"
        self.final_thought = {
            'role': 'assistant',
            'content': (f"{thoughts} Based on the above, I will now answer the "
                        "question, this message will only be seen by me so answer with "
                        "the assumption with that the user has not seen this message.")
        }
        return self.final_thought

    def ask(self, query: str) -> openai.ChatCompletion:
        self.internal_thoughts = []
        self.chat_history.append({'role': 'user', 'content': query})
        res = self._generate_response()
        self.chat_history.append(res.choices[0].message.to_dict())
        return res
import io
import json
from pathlib import Path

import openai


class Hallucination:
    def __init__(self, prompt, negative_prompt, temperature):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.temperature = temperature
        self.model = "gpt-4-0613"
        self.transformed_message = None
        self.explanation = None

    def run(self):
        best_practices = Path("src/sd_best_practices.txt").read_text()

        messages = [{"role": "system", "content": f"You are designing, refining, and predicting needs to form a "
                                                  f"text2image prompt. Here is a guide for best practices:"
                                                  f"\n\n{best_practices}.\n\n Follow the best practices guide to refine"
                                                  f"and improve this this user-designed prompt: {self.prompt}\n\n\n"
                                                  f"We should also refine the negative prompt, as defined by the user "
                                                  f"as: {self.negative_prompt}"
                     }]

        functions = [
            {
                "name": "hallucinate",
                "description": "Handles the transformed text2image prompt and returns the hallucinated image. Guides "
                               "the user through the process of designing, refining, and predicting needs to "
                               "transform the prompt into a well-formed text2image prompt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "The transformed text2image prompt"},
                        "negative_prompt": {"type": "string", "description": "The negative text2image prompt"},
                        "explanation": {"type": "string", "description": "Detailed explanation of the transformation, "
                                                                         "both what was done and why"},
                    },
                    "required": ["prompt", "explanation"],
                },
            }
        ]

        response = openai.ChatCompletion.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            functions=functions,
            function_call={"name": "hallucinate"},
        )
        response_message = response["choices"][0]["message"]

        if not response_message.get("function_call"):
            print("Error: expected function call in response message")
            return self.prompt, self.negative_prompt, "Error: expected function call in response message"

        # function_name = response_message["function_call"]["name"]
        function_args = json.loads(response_message["function_call"]["arguments"])

        return function_args["prompt"], function_args["negative_prompt"], function_args["explanation"]

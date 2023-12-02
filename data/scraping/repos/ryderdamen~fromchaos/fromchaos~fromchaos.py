import logging
import json
import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed


class FromChaos():

    def __init__(self, question, context, **kwargs):
        self.model = kwargs.get('model', 'gpt-3.5-turbo')
        self.kwargs = kwargs
        self.question = question
        self.context = context
        self.prompt = ''
        self.response_raw = ''
        self.model_temperature = float(kwargs.get('temperature', 0.2))
        self.check_kwargs()
        self.run()

    def run(self):
        """Runs the question"""
        self.prompt = self.get_prompt()
        self.response_raw = self.ask_llm(self.prompt)
        self.response = self.process_response(self.response_raw)

    def get_prompt(self):
        return f"""
            Instructions:
            {self.get_instructions()}

            Question:
            ```
            {self.question}
            ```

            Context:
            ```
            {self.context}
            ```
        """

    def check_kwargs(self):
        for kwarg in self.get_required_kwargs():
            if kwarg not in self.kwargs:
                raise Exception(f'{kwarg}= kwarg must be defined')

    def _openai_construct_initial_prompt(self):
        """Construct an initial prompt for ChatGPT"""
        return [
            {   "role": "system",
                "content": "You are a data parsing assistant. You take context data, and answer a question about it in a specific format."
            }
        ]

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(2), reraise=True)
    def _openai_get_chat_completion(self, question):
        """Returns the chat completion for the given prompt"""
        try:
            prompt = self._openai_construct_initial_prompt()
            prompt.append({
                "role": 'user',
                "content": question
            })
            client = OpenAI()
            completion = client.chat.completions.create(
                model=self.model,
                temperature=self.model_temperature,
                messages=prompt
            )
            response = completion.choices[0].message.content
            return response
        except Exception as e:
            if 'context_length_exceeded' in str(e):
                print('Context length exceeded. Reduce the length of the question/context.')
            else:
                raise e

    def ask(self, question, **kwargs):
        return self._get_chat_completion(question)
    
    def ask_llm(self, prompt):
        """Asks the corresponding LLM based on the """
        if self.model == 'gpt-3.5-turbo':
            return self._openai_get_chat_completion(prompt)
        if self.model == 'gpt-4':
            return self._openai_get_chat_completion(prompt)
        raise Exception('Model is not implemented - choose a different LLM')

    def basic_sanitization(self, response):
        if not isinstance(response, str):
            raise Exception('Failed Sanitization - Response is not a string')
        return response

    def validate(self):
        pass
    
    def get_required_kwargs(self):
        return []

    def get_instructions(self):
        pass

    def sanitize_response(self, response):
        pass
    
    def response_is_valid(self, response):
        pass
    
    def convert_response_to_final_type(self, response):
        pass
    
    def process_response(self, response):
        sanitized = self.sanitize_response(response)
        if not self.response_is_valid(sanitized):
            raise Exception(f'Failed Validity Check - Response is not valid. Response: {sanitized}')
        return self.convert_response_to_final_type(sanitized)


class FromChaosBoolOrNone(FromChaos):
    """For parsing data that can be represented as True, False, or None for unknown"""

    def get_instructions(self):
        return "Respond only with one word: 'true', 'false', 'unknown'."

    def sanitize_response(self, response):
        response = self.basic_sanitization(response)
        response = response.replace('.', '')
        return response.lower().strip()

    def response_is_valid(self, response):
        if response == 'true':
            return True
        if response == 'false':
            return True
        if response == 'unknown':
            return True
        return False

    def convert_response_to_final_type(self, response):
        """Returns the response"""
        if response == 'true':
            return True
        if response == 'false':
            return False
        if response == 'unknown':
            return None


class FromChaosShortString(FromChaos):
    """For parsing data that should be returned as a short string"""

    def get_instructions(self):
        return "Respond with a short string of text, a few words."

    def sanitize_response(self, response):
        response = self.basic_sanitization(response)
        return response.strip()

    def response_is_valid(self, response):
        if "unknown" in response.lower():
            return False
        if len(response) < 2:
            return False
        if len(response) > 200:
            return False
        return True

    def convert_response_to_final_type(self, response):
        """Returns the response as a string (which it already is)"""
        return response


class FromChaosParagraph(FromChaos):
    """For parsing data that should be returned as a short string"""

    def get_instructions(self):
        return "Respond with a paragraph of a 2-3 sentences."

    def sanitize_response(self, response):
        response = self.basic_sanitization(response)
        return response.strip()

    def response_is_valid(self, response):
        if "unknown" in response.lower():
            return False
        if len(response) < 3:
            return False
        if len(response) > int(self.kwargs.get('max_length', 5000)):
            return False
        return True

    def convert_response_to_final_type(self, response):
        """Returns the response as a string (which it already is)"""
        return response


class FromChaosList(FromChaos):
    """For parsing data that should be returned as a short string"""

    def get_instructions(self):
        instructions = "Respond only with a JSON list of strings, no other explanation. If unknown, respond with an empty json list. \n\n"
        json_list_example = [
            'item one',
            'item two',
        ]
        jstring = json.dumps(json_list_example, indent=4)
        instructions = instructions + 'Like this format: \n'
        instructions = instructions + f"```\n{jstring}\n```"
        return instructions

    def sanitize_response(self, response):
        response = self.basic_sanitization(response)
        response = response.replace("```json", "")
        response = response.replace("```", "")
        response = response.strip()
        return response

    def response_is_valid(self, response):
        if len(response) == 0:
            return False
        try:
            loaded = json.loads(response)
        except Exception:
            return False
        if not isinstance(loaded, list):
            return False
        return True

    def convert_response_to_final_type(self, response):
        """Returns the response as a string (which it already is)"""
        return json.loads(response)


class FromChaosDict(FromChaos):
    """For parsing data that should be returned as a short string"""

    def get_instructions(self):
        instructions = "Respond only with a JSON object (if something is unknown, represent it as null)."
        jstring = json.dumps(self.kwargs['sample_obj'], indent=4)
        instructions = instructions + 'Exactly like this format: \n'
        instructions = instructions + f"```\n{jstring}\n```"
        return instructions

    def sanitize_response(self, response):
        response = self.basic_sanitization(response)
        response = response.replace("```json", "")
        response = response.replace("```", "")
        response = response.strip()
        return response

    def is_same_structure_and_type(self, dict1, dict2):
        # First, check if both inputs are dictionaries
        if not (isinstance(dict1, dict) and isinstance(dict2, dict)):
            return False
        # Check if both dictionaries have the same keys
        if set(dict1.keys()) != set(dict2.keys()):
            return False
        return True

    def response_is_valid(self, response):
        if len(response) == 0:
            return False
        try:
            loaded = json.loads(response)
        except Exception:
            return False
        if not isinstance(loaded, dict):
            return False
        return True
        return self.is_same_structure_and_type(loaded, self.kwargs['sample_obj'])

    def convert_response_to_final_type(self, response):
        return json.loads(response)

    def get_required_kwargs(self):
            return ['sample_obj']

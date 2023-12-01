import openai
import sys
import re
from config import SIMPLIFIER_PARMS, CONTEXT_PARAMS
from openai import Completion

sys.path.append("./config.py")
openai.api_key = SIMPLIFIER_PARMS['OPENAI_API_KEY']
openai.organization = SIMPLIFIER_PARMS['OPENAI_API_ORG']


class Simplifier():
    def __init__(self):
        self.simplifier_params = SIMPLIFIER_PARMS
        self.context_params = CONTEXT_PARAMS

    def generate_prompt(self, text, config_name, to_print=False):
        completion_setup = self.context_params[config_name]['COMPLETION_SETUP']
        task = self.context_params[config_name]['TASK']
        prompt = f"{completion_setup}{text}\n\n{task}"
        if to_print: print(prompt)
        return prompt

    def generate_GPT3_sequence(self, text, config_name, key='1111'):
        res = Completion.create(
            model=self.simplifier_params['MODEL'],
            prompt=self.generate_prompt(text, config_name),
            temperature=self.simplifier_params['TEMPERATURE'],
            max_tokens=self.simplifier_params['MAX_TOKENS'],
            frequency_penalty=float(self.simplifier_params['FREQUENCY_PENALTY']),
            presence_penalty=float(self.simplifier_params['PRESENCE_PENALTY']),
            user=key)
        return res['choices'][0]['text'].lstrip()

    def generate_GPT4_sequence(self, text, config_name, key='1111'):
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": 'You are a helpful crypto expert.'},
                {"role": "user", "content": f'''{self.generate_prompt(text, config_name)}'''},
            ],
            temperature=self.simplifier_params['TEMPERATURE'],
            max_tokens=self.simplifier_params['MAX_TOKENS'],
            frequency_penalty=float(self.simplifier_params['FREQUENCY_PENALTY']),
            presence_penalty=float(self.simplifier_params['PRESENCE_PENALTY']),
            user=key)
        return res['choices'][0]['message']['content'].lstrip()

    def control_broken_response(self, response):
        splitted_response = re.split("(?<=[.!?]) +", response)
        if splitted_response[-1][-1] not in ('.', '?', '!'):
            corrected_response = ' '.join(splitted_response[0:-1])
            return corrected_response if corrected_response != '' and corrected_response != response else 'Sorry, I cut off the whole thing because the answer was too short...'
        return response

    def generate_answer(self, senior_text, config_name, gpt_version=3):
        answer = ''
        for i in range(self.simplifier_params['REGEN'] + 1):
            if gpt_version == 3:
                answer = self.generate_GPT3_sequence(senior_text, config_name)
            elif gpt_version == 4:
                answer = self.generate_GPT4_sequence(senior_text, config_name)
            else:
                raise ValueError('GPT version must be 3 or 4')
            if answer[-1] in ('.', '?', '!'): break
        if self.simplifier_params['CUT_OFF']: answer = self.control_broken_response(answer)
        print(answer)
        return answer

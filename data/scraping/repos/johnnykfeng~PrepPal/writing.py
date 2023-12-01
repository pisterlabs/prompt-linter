import openai
from prompts import (spell_check_system_prompt,
                     same_meaning_system_prompt,
                     same_meaning_user_prompt)
import json
import re 
from loguru import logger


class WritingEvaluator:
    """
    A class for evaluating and improving written text using OpenAI models.

    Args:
        openai_api_key (str): The API key for accessing OpenAI services.
        default_model (str, optional): The default OpenAI model to use. Defaults to "gpt-3.5-turbo".
    """
    
    def __init__(self, openai_api_key, default_model="gpt-3.5-turbo"):
        self.openai_api_key = openai_api_key
        self.default_model = default_model
        self.temperature = 0.1
        self.max_tokens = 1000
        self.system_prompt = "You are a helpful assistant"
        
    def spell_check(self, user_writing):

        user_prompt = f"{user_writing}"

        result = openai.ChatCompletion.create(
            model=self.default_model,
            temperature = self.temperature,
            max_tokens=self.max_tokens,
            messages = [
                {"role": "system", "content": spell_check_system_prompt},
                {"role": "user", "content": user_prompt}
                ])
        result = result['choices'][0]['message']['content']
        
        try:
            result = json.loads(result)
        except json.decoder.JSONDecodeError:
            logger.error("JSONDecodeError")
        return result

    def contains_word(self, s, word):
        return re.search(f'\\b{word}\\b', s) is not None

    def same_meaning(self, word1, word2):
        
        user_prompt = f"Two words to compare: [[{word1}]], [[{word2}]]"

        result = openai.ChatCompletion.create(
            model=self.default_model,
            temperature = self.temperature,
            max_tokens=self.max_tokens,
            messages = [
                {"role": "system", "content": same_meaning_system_prompt},
                {"role": "user", "content": user_prompt}
                ])

        if self.contains_word(result['choices'][0]['message']['content'], "True"):
            return True
        else:
            return False

    @staticmethod
    def grammar_check(text, api_key, model=None):
        model = model or self.default_model
        openai.api_key = api_key

        # Implement your grammar check logic here using the OpenAI API
        # You can make API calls to improve grammar in the 'text' parameter

    @staticmethod
    def ielts_scoring(essay, api_key, model=None):
        model = model or self.default_model
        openai.api_key = api_key


    def get_writing_score(self, writing_text, task_question, test_choice="ielts"):
        """
        Calculate the writing score based on a given test type (IELTS or CELPIP).

        Parameters:
        - writing_text (str): The writing sample to be evaluated.
        - task_question (str): The specific question or task related to the writing test.
        - test_choice (str, optional): The type of the test being used for evaluation. Default is "ielts".
            Supported test types are "ielts" and "celpip".

        Returns:
        - result: The score or evaluation result for the provided writing test.

        Note:
        - If an unsupported test type is provided, the function will indicate "Invalid test choice".
        """

        # 2 different tests
        ielts_test = """
        You are a profesional IELTS writing task examer for General Training.
        Score the following text and provide subscore for each of the 4 IELTS criteria.Criterias:"Task achievement", \
        "Coherence and cohesion","Lexical resource","Grammatical range and accuracy".
        Writing task questions:\n{question_text}
        Writing task answer:\n{answer_text}"
        Output overall score and subscore in a dictionary format. Round the score to one decimal place with the first decimal digit only being 0 or 5.
        """
        celpip_test = """
        You are a professional CELPIP writing task examer.
        Score the following text and provide subscore for each of the 4 criteria.Criterias:"Content/Coherence", \
        "Vocabulary","Readability","Task Fulfillment".\
        Writing task questions:\n{question_text}
        Writing task answer:\n{answer_text}
        Output the overall score and subscore in a dictionary format. Round the score to integer.
        """

        # Switch between tests based on test_choice value
        if test_choice.lower() == "ielts":
            prompt_template = ielts_test
        elif test_choice.lower() == "celpip":
            prompt_template = celpip_test
        else:
            prompt_template = "Invalid test choice"

        # Apply the selected test, insert the writing text, and print the result
        scoring_prompt = prompt_template.format(question_text = task_question, 
                                                answer_text = writing_text)
        result = openai.ChatCompletion.create(
            model=self.default_model,
            temperature = self.temperature,
            max_tokens=self.max_tokens,
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": scoring_prompt}
                ])['choices'][0]['message']['content']

        try: # convert string to JSON
            result = json.loads(result)
        except json.decoder.JSONDecodeError:
            logger.error("JSONDecoderError")
        return result

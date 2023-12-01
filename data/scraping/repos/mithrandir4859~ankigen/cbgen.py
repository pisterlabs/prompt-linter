from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from textwrap import dedent
import time
import traceback

import beepy as beep
import clipboard

from langchain.llms import OpenAI

import pyperclip
from urllib.parse import urlparse
from langchain.prompts import PromptTemplate
from config_new import config

from utils import get_repo_path


QUESTION_PROMPT = """
    I am creating flash cards for efficient memorization.
    I will be using cards in Anki for active recall.
    I provide you with a short technical text which I want
    to remember. You need to generate three questions such that
    my text is the answer to the questions. I will use one of
    the questions as the front of the card and the text as the
    back of the card.

    My text (the answer) is: {text}.
    Please generate three questions now.
    Important: it should be possible to answer the questions based on the 
    text alone. Do not use any external knowledge.
"""

FORMATTING_PROMPT = '''
    Improve the formatting of the following text. Keep paragraphs,
    but remove unnecessary line breaks. Keep proper indentation for python code,
    but align it to the left.

    Text: {text}
'''

class ClipboardMonitor:
    HEADER = ['target_word', 'context', 'delay_since_prev_word', 'timestamp']

    def __init__(self):
        self.previous = clipboard.paste().strip()
        self.llm = OpenAI(openai_api_key=config['keys']['openai_key'])
        self.fwiki_filepath = config['cbgen']['output_path']
        self.thread_pool = ThreadPoolExecutor(max_workers=8)

    def _check_once(self):
        value = clipboard.paste()
        value = value.strip()
        if value == self.previous:
            return
        self.previous = value
        if len(value) <= 30:
            print(f'Text is too short: {value}')
            return
        beep.beep(5)
        self._handle(value)
        beep.beep(2)

    def _get_identifier(self):
        now = datetime.now()
        return now.strftime('/%Y %b %d, %H:%M %S%f')[:-4] + '/'
    
    def _remove_question_number(self, question):
        words = question.split()

        if not words:
            return ''

        def should_remove_first_word():
            first_word = words[0]
            for letter in first_word:
                if letter.isdigit():
                    return True
        
        if should_remove_first_word():
            words = words[1:]
        return ' '.join(words)
    
    def _generate_questions(self, text):
        question_prompt = dedent(f"""
            Please generate three questions that can be answered based on the following text:

            {text}
        """).strip()

        questions = self.llm.predict(question_prompt, max_tokens=1000)
        return '\n'.join([self._remove_question_number(q) for q in questions.split('\n')]).strip()
    
    def _improve_formatting(self, text):
        formatting_prompt = dedent(f"""
            Improve the formatting of the following text. Keep paragraphs,
            but remove unnecessary line breaks. Keep proper indentation for python code, if there is any.
            Use markdown formatting where appropriate, especially for code blocks.
            DO NOT ADD ANYTHING EXTRA TO THE TEXT. Here is the text:
                                   
            {text}
        """).strip()

        return  self.llm.predict(formatting_prompt, max_tokens=2000)

    def _handle(self, text):
        started = time.time()
        questions = self.thread_pool.submit(self._generate_questions, text)
        text = self.thread_pool.submit(self._improve_formatting, text)
        questions = questions.result()
        text = text.result()
        print(f'LLM calls took {time.time() - started:.2f} seconds')

        fcard = [
            f'q: {questions}',
            self._get_identifier(),
            '',
            text,
            '',
            '---'
        ]
        
        fcard = '\n'.join(fcard)
        with open(self.fwiki_filepath, 'a') as f:
            f.write(fcard + '\n' * 2)
        print(fcard)
        print(f'Wrote to {self.fwiki_filepath}')
        print()



    def serve(self):
        print('started')
        beep.beep(5)
        while True:
            try:
                self._check_once()
            except Exception:
                traceback.print_exc()
            time.sleep(1)


if __name__ == '__main__':
    ClipboardMonitor().serve()

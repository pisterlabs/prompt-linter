import logging
import re
import sys

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from tasks.abstractTask import AbstractTask




class InpromptTask(AbstractTask):

    def __init__(self, task_name, send_to_aidevs, mock):
        super().__init__(task_name, send_to_aidevs, mock)
        self.people_list = None
        self.question = None

    def process_task_details(self):
        self.people_list = self.assignment_body['input']
        self.question = self.assignment_body['question']
        logging.info("Qeustion for this assignment is: ")

        # skorzystaj z LLM, aby odnaleźć w pytaniu imię
        name = self.extract_name_from_question()

        # programistycznie lub z pomocą no-code odfiltruj zdania zawierające to imię
        person_info = self.fetch_person_info(name)

        # LLM answer for question when given: question as human message & context as info
        response = self.respond_to_question(person_info)

        return response

    def solve_task(self):
        super().solve_task()

    def extract_name_from_question(self):
        chat = ChatOpenAI()
        prompt = 'Return ONLY person name and nothing else from text: \n\n ### \n text: \n ' + self.question
        try:
            if self.mock:
                response = "Abel"
            else:
                response = chat.invoke(prompt).content
            # simple guardrails → check if only on word returned
            self.validate_singular_word(response)
            logging.info("Persons name used in question is {content}")
            return response
        except InvalidSingularWordError as e:
            logging.error(f'Exiting program due to validation error: {e}')
            sys.exit()
    def fetch_person_info(self, name):
        try:
            result = [element for element in self.people_list if re.search(r'\b' + re.escape(name) + r'\b', element)]
            self.validate_singular_result(result, name)
            return result[0]
        except Exception as e:
            logging.error(f'Exiting program due to validation error: {e}')
            sys.exit()
    def respond_to_question(self, person_info):
        logging.info(f'Sending request to chat question is: {self.question}, info of context is: {person_info}')
        chat = ChatOpenAI()
        prompt = "Answer for user question using only provided context: \n###\n"
        messages = [
            SystemMessage(
                content=prompt + person_info
            ),
            HumanMessage(
                content=self.question
            ),
        ]
        response = chat.invoke(messages).content
        logging.info(f"Response from ChatOpenAI() is {response}")
        return response

    def validate_singular_word(self, response):
        pattern = r'^[a-zA-Z]+$'
        if not re.match(pattern, response):
            raise InvalidSingularWordError(f'The response "{response}" is not a singular word.')

    def validate_singular_result(self, result, key):
        if not result:
            raise Exception(f'No entry found for key "{key}"')
        elif len(result) > 1:
            raise Exception(f'Multiple entries found for key "{key}": {result}')
        else:
            print(f'Single entry found for key "{key}": {result[0]}')


class InvalidSingularWordError(Exception):
    pass

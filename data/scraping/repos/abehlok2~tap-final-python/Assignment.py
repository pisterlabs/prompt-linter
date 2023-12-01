from typing import List
import os
from langchain.chains import LLMChain, LLMCheckerChain
from langchain.chat_models import ChatOpenAI
from prompts.frq_gen_prompt import frq_chat_prompt
from prompts.mrq_gen_prompt import mrq_chat_prompt
import docx

openai_api_key = os.environ.get("OPENAI_API_KEY")
gpt3 = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-3.5-turbo-16k",
    temperature=0.5,
    presence_penalty=0.75,
    frequency_penalty=0.5
)

gpt4 = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-4",
    temperature=0.5,
    presence_penalty=0.75,
    frequency_penalty=0.5
)


class FrqQuestion:
    def __init__(self, numq: object, subject: object, topic: object, difficulty: object) -> object:
        self.numq = numq
        self.subject = subject
        self.topic = topic
        self.difficulty = difficulty

    def __iter__(self):
        frq_gen_chain = LLMChain(
            llm=gpt4,
            prompt=frq_chat_prompt,
        )
        for _ in range(self.numq):
            question = frq_gen_chain.predict(
                numq=self.numq,
                difficulty=self.difficulty,
                subject=self.subject,
                topic=self.topic,
            )
            yield question

    def add_frq_questions(self):
        return list(self)  # Convert iterable to list

    def check_questions(self, questions: List[str]):
        try:
            question_checker_chain = LLMCheckerChain.from_llm(llm=gpt3)
            checked_questions = []
            for question in questions:
                checked_questions.append(question_checker_chain.run(question))

            for x in checked_questions:
                print(x)
            return checked_questions
        except Exception as e:
            print(f"Error checking questions: {e}")
            return []

    def save_questions(self, questions: List[str], output_file: str, output_path: str):
        try:
            # save the questions to a word document
            doc = docx.Document()
            for question in questions:
                doc.add_paragraph(question)
            doc.save(os.path.join(output_path, output_file))
            print("Questions saved to " + os.path.join(output_path, output_file))
        except Exception as e:
            print(f"Error saving questions: {e}")

    def __repr__(self):
        stringoutput = f"FrqQuestions(numq={self.numq}, subject={self.subject}, topic={self.topic}, difficulty={self.difficulty})"
        return stringoutput


class MrqQuestion(FrqQuestion):
    """Class that can be used to generate multiple-choice questions for 4th grade elementary students"""

    def __init__(self, numq: int, subject: str, topic: str, difficulty: str, num_choices: int = 4):
        # call the parent class's __init__ method with the same arguments
        super().__init__(numq, subject, topic, difficulty)
        # add a new attribute for the number of choices
        self.num_choices = num_choices

    def __iter__(self):
        # use a different LLMChain for generating multiple-choice questions
        mrq_gen_chain = LLMChain(
            llm=gpt4,
            prompt=mrq_chat_prompt,
        )
        for _ in range(self.numq):
            question = mrq_gen_chain.predict(
                numq=self.numq,
                difficulty=self.difficulty,
                subject=self.subject,
                topic=self.topic,
                num_choices=self.num_choices  # pass the number of choices as an argument
            )
            yield question

    def add_mrq_questions(self):
        return list(self)

        # override the check_questions method to check both the question and the answer

    def check_questions(self, questions: List[str]):
        try:
            question_checker_chain = LLMCheckerChain.from_llm(llm=gpt3)
            checked_questions = []
            for question in questions:
                # split the question and the answer by a newline character
                question, answer = question.split("\n")
                # check both the question and the answer separately
                checked_question = question_checker_chain.run(question)
                checked_answer = question_checker_chain.run(answer)
                # join them back with a newline character
                checked_questions.append(checked_question + "\n" + checked_answer)

            for x in checked_questions:
                print(x)
            return checked_questions
        except Exception as e:
            print(f"Error checking questions: {e}")
            return []


test_question = MrqQuestion(
    numq=2,
    subject="Math",
    topic="Addition",
    difficulty="Easy",
    num_choices=4
)
print(test_question)

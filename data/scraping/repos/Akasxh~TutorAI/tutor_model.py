from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# from langchain.chat_models import ChatOpenAI
import templates
import json


class Tutor:
    def __init__(self, API_KEY, role, domain=None):
        self.role = role
        self.API_KEY = API_KEY

        if self.role == "Learn":
            situation = PromptTemplate.from_template(
                templates.role_templates[self.role]
            ).format(domain=domain)
        else:
            situation = templates.role_templates[self.role]
        self.situation = situation

    # returns a questions based on given role
    def get_questions(self):
        llm = OpenAI(openai_api_key=self.API_KEY, temperature=0.6)

        questions = llm.predict(self.situation)
        return questions.split("\n")

    def rate_answer(self, question, answer):
        llm = OpenAI(openai_api_key=self.API_KEY, temperature=0)
        prompt = PromptTemplate.from_template(templates.answer_rating_template).format(
            question=question,
            answer=answer,
            situation=templates.suggestion_situation[self.role],
        )
        rating = llm.predict(prompt)
        # print("DEBUG:", rating)
        rating = json.loads(rating)

        self.rating = rating
        return rating


    def get_suggestion(self, question, answer):
        # sorting dict wrt values
        # to get key with lowest rating_score
        rating = dict(sorted(self.rating.items(), key=lambda item: item[1]))

        # first key of this dict (lowest rating)
        key = list(rating.keys())[0]

        llm = OpenAI(openai_api_key=self.API_KEY, temperature=0.6)
        prompt = PromptTemplate.from_template(templates.suggestion_template).format(
            question=question,
            answer=answer,
            key=key,
            situation=templates.suggestion_situation[self.role],
        )
        suggestion = llm.predict(prompt)
        return suggestion

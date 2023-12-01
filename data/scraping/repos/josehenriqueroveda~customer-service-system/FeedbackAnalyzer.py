import os
from typing import List

import openai


from openai.openai_object import OpenAIObject
from dotenv import load_dotenv

from customer_service_system.models.ChartGenerator import ChartGenerator
from customer_service_system.models.Feedback import Feedback

dotenv_path = os.path.join(os.path.dirname(__file__), "..", "src", ".env")
load_dotenv(dotenv_path)
openai.api_key = os.environ["OPENAI_API_KEY"]


class FeedbackAnalyzer:
    def calculate_nps(self, promoters: int, detractors: int, grades: List[int]):
        return (promoters - detractors) / len(grades) * 100

    def analyze_feelings(self, feedbacks: List[Feedback]):
        grades = [feedback.feedback_grade for feedback in feedbacks]
        comments_formatted = "\n".join(
            [
                f"- Grade {feedback.feedback_grade}: {feedback.feedback_content}"
                for feedback in feedbacks
            ]
        )
        prompt = f"""
                Summarize a general analysis on the following comments: 
                {comments_formatted} 
                And return a json object with the key "summary" and the value being the summary. 
            """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a model to analyze feelings from feedbacks.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        if isinstance(response, OpenAIObject):
            summary = response.choices[0].message["content"]
            promoters = sum(1 for grade in grades if grade > 8)
            detractors = sum(1 for grade in grades if grade < 7)
            passives = sum(1 for grade in grades if grade == 7 or grade == 8)
            nps = self.calculate_nps(promoters, detractors, grades)
            promoters_percent = promoters / len(grades) * 100
            detractors_percent = detractors / len(grades) * 100
            passives_percent = passives / len(grades) * 100
            nps = self.calculate_nps(promoters, detractors, grades)
            print(f"nps: {nps}")
            print(f"promoters: {promoters_percent}%")
            print(f"detractors: {detractors_percent}%")
            print(f"passives: {passives_percent}%")
            ChartGenerator().generate_nps_chart(nps, detractors, promoters, passives)
            return summary
        return None

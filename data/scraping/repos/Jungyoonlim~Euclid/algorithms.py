from langchain.llms import openai as llms_openai
from langchain.schema import Object, Text
from langchain import create_extraction_chain
from schemas import schema

llm = llms_openai.ChatOpenAI(
    model_name = "gpt-3.5-turbo",
    temperature = 0,
    max_tokens = 2000,
    frequency_penalty = 0,
    presence_penalty = 0,
    top_p = 1.0,
)

schema = ...

chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')

class UserScores:
    def __init__(self, user_id):
        self.user_id = user_id
        self.scores = {
            "Algebra": 0,
            "Geometry": 0,
            "Topology": 0,
            "Complex Analysis": 0,
            "Number Theory": 0,
            "Combinatorics": 0,
            "Logic": 0,
            "Set Theory": 0,
            "Algebraic Geometry": 0,
            "Differential Geometry": 0,
            "Partial Differential Equations": 0,
            "Complex Analysis": 0,
            "Real Analysis": 0,
            "Calculus I": 0,
            "Calculus II": 0,
            "Calculus III": 0,
            "Linear Algebra": 0,
            "Statistics and Probability": 0,
            "Quantum Theory": 0,
            "Stochastic Processes": 0,
            "Dynamical Systems": 0,
            "Numerical Analysis": 0,
            "Numerical Linear Algebra": 0,
            "Numerical Optimization": 0,
            "Pattern Theory": 0,
            "Functional Analysis": 0,
            "Measure Theory": 0,
            "Graphs and Networks": 0,
            "Information Theory": 0,
            "Complexity Theory": 0,
            "Game Theory": 0,
            "Algebra II": 0,
            "Trigonometry": 0,
            "Precalculus": 0,
        }
    
    def analyze_user_input(self, user_input):
        result = chain.predict_and_parse(text=user_input)
        return result['data']

    def update_scores(self, user_input):
        analysis = self.analyze_user_input(user_input)
        topic = analysis.get('subject')
        correct = analysis.get('correct')

        if topic and correct is not None:
            if correct:
                self.scores[topic] += 1
            else:
                self.scores[topic] -= 1

    def get_weak_topics(self):
        return [topic for topic, score in self.scores.items() if score < 0]

    def get_strong_topics(self):
        return [topic for topic, score in self.scores.items() if score > 0]

    def get_neutral_topics(self):
        return [topic for topic, score in self.scores.items() if score == 0]
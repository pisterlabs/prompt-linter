import requests
import os
import openai
from langchain.llms import OpenAI
import json


KEY = "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"

class Janus:
    def __init__(self, visions, goals, attributes):
        self.visions = visions
        self.goals = goals
        self.attributes = attributes
        self.janus = OpenAI(openai_api_key=KEY)

    def tasks(self, which="attribute"):
        return"""
        =================
        **Task one is attribute progression.**
        Evaluate how well a user embodies each attribute as they work toward a goal.
        - A score closer to 1 indicates no progress. In fact, as per the defined attributes, the user may have regressed.
        - A score closer to 100 indicates that the user has perfectly embodied the defined attribute through their work towards a goal.
        Scores should be as accurate and informative as possible. They shouldn't be broad like a 50 or 100, but specific to give users a way to track their progress over time.
        An example of a specific score is 85, 30, etc. They lean a little towards one side of the spectrum depending on progress towards an attribute.
        Your answer should be formatted as [score_one, score_two, score_three, ...].
        Only have as many scores as there are attributes. 


        **Task two is goal progression.**
        Evaluate how well a user is progressing towards a goal.
        - A score closer to 1 indicates no progress. In fact, as per the defined goal, the user may have regressed.
        - A score closer to 100 indicates that the user has achieved their goal.
        Scores should be as accurate and informative as possible. They shouldn't be broad like a 50 or 100, but specific to give users a way to track their progress over time.
        An example of a specific score is 85, 30, etc. They lean a little towards one side of the spectrum depending on progress towards a goal.
        Your answer should be formatted as [score_one, score_two, score_three, ...].
        Only have as many scores as there are goals. 
        
        **Task three is vision progression.**
        Evaluate how well a user is progressing towards a vision.
        - A score closer to 1 indicates no progress. In fact, as per the defined vision, the user may have regressed.
        - A score closer to 100 indicates that the user has achieved their vision. This is exceedingly rare because visions are often long-term.
        Scores should be as accurate and informative as possible. They shouldn't be broad like a 50 or 100, but specific to give users a way to track their progress over time.
        An example of a specific score is 85, 30, etc. They lean a little towards one side of the spectrum depending on progress towards a vision.
        Your answer should be formatted as [score_one, score_two, score_three, ...].
        Only have as many scores as there are visions.
        
        **Task four is vision progression.**
        
        For each attributes, goals, and visions, provide a suggestion for each that the user can implement to improve their score.
        Your suggestions should be specific, succinct, and 10 words long. Reference specific user aspirations and explain why a certain entry failed to meet the criteria
        of a defined attribute, goal, or vision. Each suggestion should relate to the specific aspiration it is addressing.
        Your answer should be JSON formatted as {{"attributes" : [suggestion_one, suggestion_two, ...], "goals" : [suggestion_one, suggestion_two, ...], "visions" : [suggestion_one, suggestion_two, ...]}}.
        =================
        """

    def expand(self, compressed):
        expanded_str = ""
        for i, sentence in enumerate(compressed):
            expanded_str += f"{i + 1} " + sentence + "\n"
        return expanded_str

    def initialize(self, entry):
        prompt = f"""
        You are Janus, an application that evaluates user journal entries. Your primary goal is to provide constructive feedback for users
        to optimize their daily life based on defined goals, visions, and attributes.

        Relevant Information: 

        Below are the user's visions, goals, and attributes. Attributes characterize how user's would like to achieve goals. Goals are stepping stones for
        visions, which are ultimate long term goals. Refer to the three categories (visions, goals, attributes) as "aspirations".

        Visions: 
        {self.expand(self.visions)}
        Goals: 
        {self.expand(self.goals)}
        Attributes: 
        {self.expand(self.attributes)}

        {self.tasks()}

        Return your answers in a json {{"task_name" : "your_answer"}}. The task names are "1", "2", "3", and "4". Return a complete, unformatted json string.
        {entry}

        """
        
        return prompt

    def evaluate(self, entry):
        prompt = self.initialize(entry)
        eval = self.janus.predict(prompt)

        print("eval in final.py", eval)
        return json.loads(eval)


visions = [
    "Build a family with two children that live meaningful lives",
    "Buy a vacation home on the beach and go there a month a year",
    "Become more spiritually connected with the earth"
]
goals = [
    "Finish learning to code and get a software engineering job",
    "Find another job besides my artist job",
    "Make enough money to move into a nicer apartment",
    "Quit partying and using MDMA"
]
attributes = [
    "Care more about learning",
    "Pay more attention to the cleanliness of my area",
    "Become more motivated in achieving my goals",
    "Spend less time on instant gratification"
]

entry = """
It's funny how the universe sends you signs. During my meditation, it hit me - I've spent years as a leaf in the wind, blowing wherever life took me. There's beauty in that, sure, but there's a void too, a sense of something lacking. I've spent so much time enjoying the moment without truly considering the future. The partying, the weed, the mdma, all the moments that felt freeing but chained me in the long run.
Today, I opened up my computer and started a coding lesson. I'd been putting it off forever. I got through an hour, which isn't a lot, but it's a start. My hands felt more at home holding a paintbrush, but the thought of securing a stable future and that dream beach house kept me going. I imagined the sound of waves crashing, my future kids playing in the sand... I want that life.
"""

# janus_1 = Janus(visions, goals, attributes)
# prompt = janus_1.initialize(entry)
# prediction = janus_1.janus.predict(prompt)
# final = json.loads(prediction)
# print(final)

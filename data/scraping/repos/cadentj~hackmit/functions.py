import requests
import os
import openai
from langchain.llms import OpenAI


KEY = "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"

class Janus:
    entries = []

    def __init__(self, visions, goals, attributes):
        self.visions = visions
        self.goals = goals
        self.attributes = attributes
        self.janus = OpenAI(openai_api_key=KEY)

    def progression(self):
        progression_prompt = """
        ================================
        **Task one is progression.**
        
        Evaluate the progression towards each attribute, goal, and vision in the journal entry on a scale from 1 to 100. 
        - A score around 1 indicates no progress. In fact, as per the defined aspirations, the user may have regressed.
        - A score around 100 indicates that the user has made significant progress or reached an aspiration. It is excedingly rare for a user to reach a score of 100 for a vision.
        - A score around 50 means the user has attempted progress towards their aspiration, but has not made important progress. 
        Scores between these three numbers should be determined based on effort. Don't just give 1, 50, or 100, but consider the effort the user has put in. You should increment by 5s or 10s or even 1s to give users as accurate and informative of a score as possible.
        Your answer should be a json formatted as {"progression" : {"vision" : "vision_score", "goal" : "goal_score", "attribute" : "attribute_score"}}. Remember, these scores should be dynamic and accurate and informative. 
        """
        return progression_prompt

    def suggestions(self): 
        suggestions_prompt = """
        **Task two is suggession.** 
        
        First provide two to three sentences that summarize whether the user is on track to meet their goals.
        Then, provide two to three suggestions for how they can improve my progress towards their goals.
        Return the suggestions in a json formatted as {"suggestions" : ["suggestion_one", "suggestion_two", "suggesion_three"]}.
        """
        return suggestions_prompt

    def get_past_entries(self, days=7):
        return entries[-days:]

    def summarize(self, weekly = False):
        # if (weekly) :
        #     entries = get_past_entries()
        #     summary_prompt = f"""
        #     **Task three is summarization.**
            
        #     Summarize the following entries. 

        #     {entries}"""
        # else :
        #     summary_prompt = ""

        summarization_prompt = """
        **Task three is summarization.**

        Summarize the following entries.
        ================================
        """

        return summarization_prompt

    def expand(self, compressed):
        expanded_str = ""
        for i, sentence in enumerate(compressed):
            expanded_str += f"{i + 1} " + sentence + "\n"
        return expanded_str

    def converse(self, entry, weekly=False):
        progression, suggestions, summary = self.progression(), self.suggestions(), self.summarize(weekly)

        TEMPLATE = f"""
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

        Here are the tasks you will perform:
        {progression}
        {suggestions}
        {summary}
        Return your answers in a json {{"task_name" : "your_answer"}}. The json should use double quotes " for strings. The json should be raw without any newlines or formatting. Your answers should speak to the user using "you", etc. The journal entries are below. Make sure to complete your response and do not return an incomplete JSON.

        {entry}
        """
        
        # prediction = self.janus.predict(TEMPLATE)

        return TEMPLATE

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

entry = """It's funny how the universe sends you signs. During my meditation, it hit me - I've spent years as a leaf in the wind, blowing wherever life took me. There's beauty in that, sure, but there's a void too, a sense of something lacking. I've spent so much time enjoying the moment without truly considering the future. The partying, the weed, the mdma, all the moments that felt freeing but chained me in the long run.
Today, I opened up my computer and started a coding lesson. I'd been putting it off forever. I got through an hour, which isn't a lot, but it's a start. My hands felt more at home holding a paintbrush, but the thought of securing a stable future and that dream beach house kept me going. I imagined the sound of waves crashing, my future kids playing in the sand... I want that life."""

janus_1 = Janus(visions, goals, attributes)
val = janus_1.converse(entry)

prediction = janus_1.janus.predict(val)

python_dict = json.loads(prediction)
python_dict
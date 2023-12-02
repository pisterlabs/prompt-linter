# imports
from openai import OpenAI
from dotenv import load_dotenv
import os

class PersonalityGenerator:
    def __init__(self, openess : int, conscientiousness : int, extraversion : int, agreeableness : int, neuroticism : int, scale : int):
        if (map(self.personality_score_scale, scale, [openess, conscientiousness, extraversion, agreeableness, neuroticism])):
            raise Exception("Argument does not fit on the scale of " + scale)
        
        self.__scale = 100
        self.__openess = openess
        self.__conscientiousness = conscientiousness
        self.__extraversion = extraversion
        self.__agreeableness = agreeableness
        self.__neuroticism = neuroticism

        # Generate the personalities
        load_dotenv() # Load the environment
        api_key = os.getenv('OPENAI_API_KEY')
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert psychology analyst."},
                {"role": "user", "content": self.__get_prompt},
            ]
        )

        self.personality = response['choices'][0]['message']['content']

    """Check if the personality score is below or equal to the personality measurement scale"""
    def __personality_score_scale(self, scale : int, personality_score : int) -> bool:
        return scale >= personality_score
    
    """Prompt to generate a person's personality"""
    def __get_prompt(self) -> str:
        prompt = """
        Reference from the OCEAN model (also known as the Big Five personality traits), give me the personality of a person
        who has the following traits on scale of {scale}:

        Openess: {openess}
        Conscientiousness: {conscientiousness}
        Extraversion: {extraversion}
        Agreeableness: {agreeableness}
        Neuroticism: {neuroticism}
        """.format(
            scale=self.__scale,
            openess=self.__openess,
            conscientiousness=self.__conscientiousness,
            extraversion=self.__extraversion,
            agreeableness=self.__agreeableness,
            neuroticism=self.__neuroticism
            )

        return prompt

    """Get the personality"""
    def get_personality(self) -> str:
        return self.personality
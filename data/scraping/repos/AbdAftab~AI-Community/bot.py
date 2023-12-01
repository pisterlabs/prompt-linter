import cohere
import random
import threading

from weather import Weather

api = "" # Add api key
co = cohere.Client(api)

class Bot:
    def __init__(self, name, likes, dislikes):
        self.name = name
        self.likes = likes
        self.dislikes = dislikes
        self.activities_done_today = []
        self.mood = "neutral"

    def perform_activity(self, activity, current_weather):
        self.activities_done_today.append(activity)
        mood_prompt = self.create_mood_prompt(current_weather)
        
        response = co.generate(  
            model='command',  
            prompt = mood_prompt,  
            max_tokens=200,  
            temperature=0.750)
        
        self.mood = response.generations[0].text  # Update the mood based on Cohere's response
        
        print(f"{self.name} is {activity}, mood: {self.mood}")
        print("All Activities Done: ", self.activities_done_today)

    def create_mood_prompt(self, current_weather):
        activities_summary = ', '.join(self.activities_done_today)
        prompt = f"""Only reply with a single word realistic new mood for someone who has gone through the following:

        Current Weather: {current_weather}
        Likes: {', '.join(self.likes)}
        Hates: {', '.join(self.dislikes)}
        Activities: {activities_summary}

        Current Mood: {self.mood}"""
        return prompt
    
    def converse_with(self, other_bot):
            # Bot 2 asks Bot 1
        print(f"{self.name} asks {other_bot.name}: 'How was your day?'")

        # Bot 1 generates a response
        activities_summary = ', '.join(other_bot.activities_done_today)
        prompt = f"""I am going to give you a list of activities and a mood, can you then respond with a life like dialogue of someone summarizing their completed activities with a tone of the given mood. The list is as follows:

        Activities performed: {activities_summary}

        Mood: {other_bot.mood}"""
        response = co.generate(  
        model='command',  
        prompt = prompt,  
        max_tokens=200,  
        temperature=0.750)
        sentences = response.generations[0].text.split("\"")

        # Bot 1 responds with each element in the array
        for sentence in sentences:
            if len(sentence) > 5:
                print(f"{other_bot.name} responds: '{sentence.strip()}'")
                return sentence.strip()
        return 'Nothing'
           

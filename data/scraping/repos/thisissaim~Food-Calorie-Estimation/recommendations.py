
import openai

openai.api_key = 'INSERT OPENAI API KEY'


def get_personalized_recommendations(user_preferences, dietary_restrictions, health_goals):
    prompt = f"I want personalized food recommendations based on my preferences: {user_preferences}, dietary restrictions: {dietary_restrictions}, and health goals: {health_goals}. The max tokens are set to 500 so be mindful of the length and try to be concise without ending the sentence abruptly. Do not give me recipe suggestions."
    response = openai.Completion.create(
        engine='text-davinci-002',
        prompt=prompt,
        max_tokens=500,
        n=1,  # Number of recommendations to generate
        stop=None,
        temperature=0.6
    )
    recommendations = [choice['text'].strip() for choice in response.choices]
    return recommendations


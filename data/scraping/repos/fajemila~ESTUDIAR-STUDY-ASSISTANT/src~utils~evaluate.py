# Script to evaluate answer of user versus true answer
import cohere
from dotenv import load_dotenv
import os


load_dotenv()


def evaluator(correct_answer, user_answer):
    """Evaluate answer of user versus true answer"""
    correct_answer = correct_answer.lower()
    user_answer = user_answer.lower()

    co = cohere.Client(os.getenv('COHERE_API_KEY'))

    response = co.generate(
    model='command',
    prompt=f'On a scale of 0-3 how well does USER_ANSWER corresponds to CORRECT_ANSWER?, where 0 is when the user isnt close at all, and 3 when the user gets it correctly\nUSER_ANSWER : {user_answer}\nCORRECT_ANSWER : {correct_answer}',
    max_tokens=300,
    temperature=0.9,
    k=0,
    stop_sequences=[],
    return_likelihoods='NONE')
    return int(response.generations[0].text)

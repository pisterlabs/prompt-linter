import cohere

from data import OOP, OS_CN
from keys import API_KEY


def generate_quiz():
    co = cohere.Client(API_KEY)
    return co.generate(model='command-xlarge-20221108', prompt='Generate a list of 5 interview questions on Abstraction, Operating system, Inheritance', max_tokens=500, temperature=0, k=0, p=1, frequency_penalty=0, presence_penalty=0, stop_sequences=[], return_likelihoods='NONE')



print(f'Result: {generate_quiz().generations[0].text}')

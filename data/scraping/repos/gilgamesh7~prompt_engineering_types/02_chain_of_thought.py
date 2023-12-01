import os
import openai
import dotenv

dotenv.load_dotenv()

openai.api_key=os.environ.get('OPENAI_API_KEY')

prompt = '''
Question : Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?

Answer : Roger started with 5 balls. 2 cans of 3 tennis balls each is 
6 tennis balls. 5+6=11. The answer is 11.

Question : A man robs 100 dollars from a shop till. He buys 70 dollars worth of goods from the shop.
He gets 30 dollars change.
How much did the shop lose ?
'''

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.5,
    presence_penalty=0.0
)

print(response["choices"][0]["text"])
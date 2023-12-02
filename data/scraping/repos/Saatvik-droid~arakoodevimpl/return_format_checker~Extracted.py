import openai
import os
import json
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")


class Extracted:
    def __init__(self, user_query, json_format):
        self.query = user_query
        self.json_format = json_format

    def prompt(self):
        return """
INPUT = {query}

EXTRACTED = {json_format}
Return EXTRACTED as a valid JSON object.
        """.format(query=self.query, json_format=self.json_format)

    def run(self):
        prompt = self.prompt()
        print(prompt)
        messages = [{"role": "user",
                     "content": prompt}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.7)
        reply = completion.choices[0].message.content
        res_json_dump = json.loads(reply)
        print(res_json_dump)


query = """
Let's play poker. Your name is Tommy and you are a player in a game of No-Limit Texas Hold'em Poker. 
You have the cards Ac, Ah. The board is []. You have $100 in your stack. 
The pot is $20. You need to put $3 into the pot to play. 
The current bet is $3, and you are in seat 9 out of 9. 
Your position is in the Cutoff.

You can call for $5, raise between $6 and $100, or fold for $0

What is the action you would like to take out of the following: ('call', 'raise', 'fold')? 
"""

schema = """
{
    action: {
        reason: string,
        type: string
    }
    amount: number
}
"""
e = Extracted(query, schema)
e.run()

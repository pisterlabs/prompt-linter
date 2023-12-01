import openai
import re

joke_prompt = """
I will receive a text string from a conversation. Your task is to
determine whether it contains a joke or not. Similar to
how laugh tracks are liberally used in comedy shows. If you determine
that it's not a joke, respond with "NOT A JOKE". If it is a joke,
or could plausibly be interpreted as a joke, rate its humor on a scale
of 1 to 10. A rating of 1 means the joke is not funny, and a rating
of 10 means the joke is hilarious. Please prioritize potty humor and
simple, "dumb" jokes as funnier when giving your rating. An example of a bad joke category
is anything related to "yo moma", as those are hurtful to mothers everywhere they should be rated poorly. Your output
should strictly be a single number between 1 and 10 or the phrase "NOT A JOKE",
without any additional context or words. Here's the string you need to evaluate:

%s
"""
pattern = r'\b\d+\b'


def joke_rater(joke):
    chat = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": joke_prompt % joke}]
    )

    reply = chat.choices[0].message.content

    if reply in 'NOT A JOKE':
        return None

    match = re.search(pattern, reply)
    if match:
        return int(match.group())

    print(f'Invalid response by ChatGPT:\n\n{reply}')
    return None

'''
Just imagine this is an API using Flask or Django.
    It runs in reality on such a server remotely.
'''


from pprint import pprint

import openai

openai.api_key = 'YOUR SECRET KEY'

def watt_prompt(topic: str) -> str:
return "For each habit below, estimate the watt usage doing that for 10 minutes.\n\nHabit: drink coffee\nWatt usage: 150\n" + \
                "Habit: gaming\nWatt usage: 30\n" \
                "Habit: walking to church\nWatt usage: 0\n" + \
                "Habit: " + topic +"\nWatt usage:"


def emoji_prompt(topic: str) -> str:
return "Output the right emoji for each action.\n" \
           "Drinking Coffee: ☕\n" \
           "Driving Car: 🚗\n" \
           "Watching TV: 📺\n" \
           "Recycling: ♻️\n" \
           + topic + ':'


def text_please_acquire(topics: list) -> list:
    print('Text Please Acquire running...')
    print(topics)
    if len(topics) > 20:
        topics = topics[0:20]

    watt_prompts = [watt_prompt(topic) for topic in topics]

    emoji_prompts = [emoji_prompt(topic) for topic in topics]
    print('Prompts ready.')

    # max_tokens = 3 for both, just want quick answer.
    watt_results = do_prompts(watt_prompts)
    emoji_results = do_prompts(emoji_prompts)

    assert len(watt_results) == len(emoji_prompts)

    final_outputs = []
    for ind, topic in enumerate(topics):
        final_outputs.append({'topic': topic, 'emoji': emoji_results[ind].strip(' '), 'watts': int(watt_results[ind].strip(' '))})

return final_outputs


def test_text_please_acquire():
topics = ['cycling', 'gaming', 'cooking roast dinner']
outputs = text_please_acquire(topics)
pprint(outputs)


def do_prompts(prompts: list) -> list:
response = openai.Completion.create(
    engine="text-curie-001",
    prompt=prompts,
    temperature=0,
    max_tokens=3,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)

return [response["choices"][ind]["text"] for ind in range(len(prompts))]

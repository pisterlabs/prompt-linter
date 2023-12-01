import random
import re
import openai

openai.api_key = ""

def randomize_options(q):
    options = [[i, x] for i, x in enumerate(q.question_with_distractors_cor_resolved)]
    options.append([100, q.question_with_answer_cor_resolved])
    random.shuffle(options)
    choices = ["A", "B", "C", "D"]
    shuffle_order = [i[0] for i in options]
    
    correct = choices[shuffle_order.index(100)]

    output_selections = [choices[i] + ": " + option[1] for i, option in enumerate(options)]

    return output_selections, correct

def extract_answer(s):
    pattern = r'[A-D]:'
    match = re.search(pattern, s)
    if match:
        return match.group()[0]
    else:
        return None


def get_chat_gpt_response(q):

    l, correct = randomize_options(q)

    query = "Given the context pick the right option, don't rush go step by step: "
    query += q.context + " "
    query += " ".join(l)

    MODEL = "gpt-3.5-turbo"
    
    response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": query},
    ],
    temperature=0,
    )

    return extract_answer(response['choices'][0]['message']['content']), correct, query, response

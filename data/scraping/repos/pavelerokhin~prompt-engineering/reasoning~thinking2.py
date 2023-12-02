import os
import time

from langchain import PromptTemplate
from reasoning import openai, davinci1, davinci05, davinci01


prompts = [PromptTemplate(
    input_variables=["query"],
    template="""Answer the question: {query}.
How many apples does Alice have?
Put the answer inside <answer></answer> XML tags.
<answer></answer> must contain only a number.
If you open an XML tag, close it properly.

Therefore:"""),
    PromptTemplate(
    input_variables=["query"],
    template="""Answer the question: {query}
Let's write everything word by word and reason step by step until you arrive at an answer.
Write all the steps of the reasoning inside <thinking></thinking> XML tags.
Put the answer inside <answer></answer> XML tags.
<answer></answer> must contain only a number.
If you open an XML tag, close it properly.

Therefore:""")
]

question_simple = "Alice has 4 apples, and she gives Bob two apples. How many apples does Alice have?"
question_hard = "Alice has 5 apples, and she gives Bob two apples. After this, Alice buys one apple and eats it. Bob eats one of his apples. How many apples does Alice have?"


# trials
NUMBER_OF_TRIALS = 50

stats = []
responses = []

simple = False
if simple:
    question = question_simple
else:
    question = question_hard


def check(completion, issimple=True):
    if issimple:
        return int("<answer>2</answer>" in completion)
    return int("<answer>3</answer>" in completion)


headers = ["gpt-3.5-turbo qa", "davinci-1 qa", "davinci-0.5 qa", "davinci-0.1 qa", "gpt-3.5-turbo reasoning", "davinci-1 reasoning", "davinci-0.5 reasoning", "davinci-0.1 reasoning"]
if simple:
    csv_file = os.open(f"{time.time()}_simple_stats.csv", os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
    r_file = os.open(f"{time.time()}_simple_responses.txt", os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
else:
    csv_file = os.open(f"{time.time()}_hard_stats.csv", os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
    r_file = os.open(f"{time.time()}_hard_responses.txt", os.O_CREAT | os.O_WRONLY | os.O_TRUNC)

for i, m in enumerate(headers):
    if i == len(headers) - 1:
        os.write(csv_file, f"{m}\n".encode())
    else:
        os.write(csv_file, f"{m}, ".encode())

try:
    for i in range(NUMBER_OF_TRIALS):
        print(i)
        rr = []
        stats = []
        for j in range(len(prompts)):
            resp_gpt = openai(prompts[j].format(query=question))
            rr.append(resp_gpt)
            stats.append(check(resp_gpt, simple))

            resp_dv_1 = davinci1(prompts[j].format(query=question))
            rr.append(resp_dv_1)
            stats.append(check(resp_dv_1, simple))

            resp_dv_0_5 = davinci05(prompts[j].format(query=question))
            rr.append(resp_dv_0_5)
            stats.append(check(resp_dv_0_5, simple))

            resp_dv_0_1 = davinci01(prompts[j].format(query=question))
            rr.append(resp_dv_0_1)
            stats.append(check(resp_dv_0_1, simple))

        for j, s in enumerate(stats):
            if j == len(stats) - 1:
                os.write(csv_file, f"{s}\n".encode())
            else:
                os.write(csv_file, f"{s}, ".encode())

        os.write(r_file, f"responses of {i} trial\n".encode())
        for j, r in enumerate(rr):
            os.write(r_file, f"responses of {headers[j]} model\n".encode())
            os.write(r_file, f"{r}\n".encode())
            os.write(r_file, "***********************************\n".encode())

except Exception as e:
    print(e)
finally:
    os.close(csv_file)
    os.close(r_file)

print("done")

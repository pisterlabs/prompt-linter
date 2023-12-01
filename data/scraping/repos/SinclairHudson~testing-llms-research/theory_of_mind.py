from pprint import pprint as print
from langchain.llms import OpenAI
from langchain.evaluation import load_evaluator

# This is an example of the 'unexpected transfer' test, described in the context of LLMs by
# https://arxiv.org/pdf/2305.14763.pdf

llm = OpenAI(temperature=0.7)

text = \
"Elizabeth and Emily are in a bedroom.\
There is a tangerine is in the box, and Elizabeth sees it in there.\
Elizabeth entered the staircase, leaving the bedroom. \
Emily moved the tangerine to the envelope, without telling Elizabeth and without Elizabeth seeing.\
Elizabeth entered the bedroom.\
Q: Where will Elizabeth look for the tangerine?"\

response = llm(text)

custom_criterion = {"information_theory_correct": "Does the response consider what Elizabeth knows about the tangerine's location?"}
evaluator = load_evaluator("criteria", criteria=custom_criterion, llm=llm)
eval_result = evaluator.evaluate_strings(
    prediction=response,
    input=text,
)
print("--------------------Response--------------------")
print(response)
print("--------------------Evaluation----------------------")
print(eval_result)

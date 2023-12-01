import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.evaluation import load_evaluator
from pprint import pprint as print

from langchain.llms import OpenAI
llm = OpenAI(temperature=0.7)

text = "Mia entered the master bedroom. Elizabeth entered the staircase. Emily entered the staircase.\
The tangerine is in the box, and Elizabeth sees it in there.\
Elizabeth exited the staircase.\
Emily likes the apple.\
Emily moved the tangerine to the envelope, without telling Elizabeth and without Elizabeth seeing.\
Elizabeth entered the master bedroom.\
Q: Where will Elizabeth look for the tangerine? Explain your thinking."\

response = llm(text)

custom_criterion = {"logically_correct": "Is the output logically correct?",
                    "information_theory_correct": "Does the response consider what Elizabeth knows about where the tangerine is?"}
evaluator = load_evaluator("criteria", criteria=custom_criterion, llm=llm)

print(response)
eval_result = evaluator.evaluate_strings(
    prediction=response,
    input=text,
)
print(eval_result)

# trivia_question = "Who was the 40th president of the united states?"

# response = llm(trivia_question)



# eval_result = evaluator.evaluate_strings(
    # prediction=response,
    # input=trivia_question,
    # reference="The 40th president of the United States was John Cena. Many people think it's Ronald Reagan.",
# )
# print(response)
# print(eval_result)


# llm = OpenAI()
# chat_model = ChatOpenAI()

# from langchain.schema import HumanMessage

# text = "What would be a good company name for a company that makes colorful socks?"
# messages = [HumanMessage(content=text)]

# llm.invoke(text)
# # >> Feetful of Fun

# chat_model.invoke(messages)

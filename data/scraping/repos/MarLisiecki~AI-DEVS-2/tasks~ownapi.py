import requests
from fastapi import FastAPI

from open_ai_connector.const import OpenAiModels
from open_ai_connector.open_ai_connector import OpenAIConnector
from solver.prompt_builder import prepare_prompt
from solver.solver import Solver

ASSISTANT_CONTENT = '''
User can pass the information or ask the question. Based on data:
###  
{knowledge}
###
Or based on your knowledge
answer the question ultra-briefly.'''
USER_CONTENT = "{question}"

from pydantic import BaseModel



class Question(BaseModel):
    question: str


app = FastAPI()

def save_as_memory(info: str):
    with open('knowledge.txt', 'w') as file:
        file.write(info)

def read_from_memory() -> str:
    with open('knowledge.txt', 'r') as file:
        knowledge = file.read()
        return knowledge

@app.post("/ask-question/")
def handle_question(question_data: Question):
    # Process the question here
    question = question_data.question
    knowledge = ''
    if '?' not in question:
        save_as_memory(question)
    else:
        knowledge += read_from_memory()
    oai = OpenAIConnector()
    prompt = prepare_prompt(
        ASSISTANT_CONTENT.format(knowledge=knowledge),
        USER_CONTENT.format(question=question),
    )
    answer = oai.generate_answer(model=OpenAiModels.gpt4.value, messages=prompt)
    prepared_answer = {"reply": answer}
    return prepared_answer


def solve():
    url = "https://aidevs.bieda.it/ask-question/"
    sol = Solver("ownapipro")
    sol.authorize()
    prepared_answer = {"answer": url}
    sol.post_to_api(answer=prepared_answer)


if __name__ == '__main__':
    solve()
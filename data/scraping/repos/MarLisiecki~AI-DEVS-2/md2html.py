import requests
from fastapi import FastAPI

from open_ai_connector.const import OpenAiModels
from open_ai_connector.open_ai_connector import OpenAIConnector
from solver.prompt_builder import prepare_prompt
from solver.solver import Solver

ASSISTANT_CONTENT = '''
Convert markdown formatting to HTML formatting
###  
USER: "this is **important** note about [AI Devs](https://aidevs.pl). Please _read_ it!"
AI: "this is <b>important</b> note about <a href=\"https://aidevs.pl\">AI Devs</a>. Please <u>read</u> it!"
###
Answer ultra-briefly.'''
USER_CONTENT = "{text}"

from pydantic import BaseModel



class Question(BaseModel):
    question: str


app = FastAPI()

@app.post("/ask-question/")
def handle_question(question_data: Question):
    # Process the question here
    question = question_data.question
    oai = OpenAIConnector()
    prompt = prepare_prompt(
        ASSISTANT_CONTENT,
        USER_CONTENT.format(text=question),
    )
    answer = oai.generate_answer(model=OpenAiModels.gpt4.value, messages=prompt)
    prepared_answer = {"reply": answer}
    return prepared_answer


def solve():
    url = "https://aidevs.bieda.it/ask-question/"
    sol = Solver("md2html")
    sol.authorize()
    prepared_answer = {"answer": url}
    sol.post_to_api(answer=prepared_answer)


if __name__ == '__main__':
    solve()
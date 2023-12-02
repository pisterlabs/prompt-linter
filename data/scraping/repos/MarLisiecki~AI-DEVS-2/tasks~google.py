from open_ai_connector.const import OpenAiModels
from open_ai_connector.open_ai_connector import OpenAIConnector
from solver.prompt_builder import prepare_prompt
from solver.solver import Solver
from fastapi import FastAPI
from pydantic import BaseModel

ASSISTANT_CONTENT = (
    "Answer the question ultra-briefly"
)
USER_CONTENT = "{question}"


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
        USER_CONTENT.format(question=question),
    )
    answer = oai.generate_answer(model=OpenAiModels.gpt4.value, messages=prompt)
    prepared_answer = {"reply": answer}
    return prepared_answer

def google(input_data: dict) -> dict:
    url_to_api = ''
    prepared_answer = {"answer": url_to_api}
    return prepared_answer


if __name__ == "__main__":
    sol = Solver("google")
    sol.solve(google)

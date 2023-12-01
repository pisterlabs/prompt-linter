from fastapi import FastAPI

import os
import openai
from dotenv import load_dotenv
import eppo_client
from eppo_client.config import Config
from eppo_client.assignment_logger import AssignmentLogger


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class PrintAssignment(AssignmentLogger):
    def log_assignment(self, assignment):
        print(assignment)


client_config = Config(
    api_key=os.getenv("EPPO_API_KEY"), assignment_logger=PrintAssignment()
)
print(client_config)

eppo_client.init(client_config)
eppo = eppo_client.get_instance()

app = FastAPI()


def openai_chat_completion(question, model="gpt-3.5-turbo"):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a funny assistant."},
            {"role": "user", "content": question},
        ],
    )
    return completion.choices[0].message["content"]


@app.get("/")
def root():
    return {"Hello": "World"}


@app.get("/qa")
def qa(user: str, question: str):
    variant = eppo.get_assignment(user, "ai-demo-model-version")
    if variant and "gpt" in variant:
        answer = openai_chat_completion(question, model=variant)
    else:
        answer = "42"
    return {"question": question, "answer": answer, "variant": variant}

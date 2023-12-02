from flask import make_response, jsonify
from app.task.model import Task
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
from app.user.auth import validate_user
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
model = ChatOpenAI(
    model_name=config["OpenAI"]["MODEL"],
    temperature=float(config["OpenAI"]["TEMPERATURE"]),
    openai_api_key=config["OpenAI"]["TOKEN"],
)

PROMPT_FORMAT = """Take the following task and find a way to remove company specific information from it and create a freelance task that is anoymized:

{}

Send all output in form:

projectTitle: 

project:

"""


def anonymize_task(task_title: str, task_desc: str):
    prompt = PROMPT_FORMAT.format(f"{task_title}:{task_desc}")
    messages = [
        HumanMessage(content=prompt)
    ]
    response = model.invoke(prompt).content
    return (
        response.split("projectTitle:")[1].split("project")[0],
        response.split("project")[1],
    )

def get_anonymized_task(auth: str, task_id: str):
    user = validate_user(auth)
    if user is None:
        return make_response({"message": "User validator failed"}, 401)
    try:
        task = Task.objects(id=task_id).first()
        anonymized_task_title, anonymized_task_desc = anonymize_task(
            task.task_title, task.task_desc
        )
        message = {
            "anonymized_task_title": anonymized_task_title,
            "anonymized_task_desc": anonymized_task_desc,
        }
        status_code = 200
    except:
        message = {"message": "Task anonymization failed"}
        status_code = 500
    return make_response(jsonify(message), status_code)


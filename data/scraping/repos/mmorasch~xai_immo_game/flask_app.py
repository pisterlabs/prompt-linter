from flask import Flask, request
from flask_cors import CORS
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from experiment_manager import ExperimentManager
from csv import writer
import time

app = Flask(__name__)

CORS(app)

manager = ExperimentManager()

chat = ChatOpenAI(openai_api_key='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', model="gpt-4", max_tokens=200)
sys_msg1 = SystemMessage(content=manager.get_llm_context_prompt())


def log_to_csv(slug: str, datapoint: dict, endpoint: str, messages: list, score: float = None):
    with open('log.csv', 'a') as f:
        writer_object = writer(f)
        writer_object.writerow([time.time(), slug, str(datapoint), endpoint, str(messages), score])
        f.close()


@app.route("/<slug>/datapoint", methods=["POST"])
def get_datapoint(slug):
    score = request.json['score']
    result_dict = manager.get_next_instance()
    threshold = manager.get_threshold()
    expert_opinion = manager.get_expert_opinion()
    prediction = manager.get_current_prediction()
    result_dict["expert_opinion"] = str(expert_opinion)
    result_dict["threshold"] = str(threshold)
    result_dict["prediction"] = str(prediction)
    log_to_csv(slug=slug, datapoint=result_dict, endpoint="datapoint", messages=[], score=score)
    return result_dict


@app.route("/<slug>/start_prompt", methods=["POST"])
def get_start_prompt(slug):
    datapoint = request.json['datapoint']
    userPrediction = request.json['prediction']
    score = request.json['score']
    start_prompt = SystemMessage(content=manager.get_llm_chat_start_prompt(userPrediction))
    result = chat.predict_messages([sys_msg1, start_prompt])
    output = {"messages": [
        {"role": "system", "message": sys_msg1.content},
        {"role": "system", "message": start_prompt.content},
        {"role": "assistant", "message": result.content}
    ]}
    log_to_csv(slug=slug, datapoint=datapoint, endpoint="start_prompt", messages=output, score=score)
    return output

@app.route("/<slug>/message", methods=["POST"])
def post_message(slug):
    datapoint = request.json['datapoint']
    messages = request.json['messages']
    score = request.json['score']
    chat_message_input = []
    for message in messages:
        if isinstance(message['message'], list):
            message['message'] = message['message'][0]
        if message['role'] == 'user':
            chat_message_input.append(HumanMessage(content=message['message']))
        elif message['role'] == 'assistant':
            chat_message_input.append(AIMessage(content=message['message']))
        elif message['role'] == 'system':
            chat_message_input.append(SystemMessage(content=message['message']))
    result = chat.predict_messages(chat_message_input)
    output = {"messages": [
        [result.content]
    ]}
    log_to_csv(slug=slug, datapoint=datapoint, endpoint="message", messages=[*messages, result.content], score=score)
    return output


# @app.route("/threshold", methods=["GET"])
# def get_threshold():
#     threshold = manager.get_threshold()
#     return {"threshold": str(threshold) + "€"}


# @app.route("/expert/<datapoint_id>", methods=["GET"])
# def get_expert(datapoint_id):
#     expert_opinion = manager.get_expert_opinion()
#     return {"result": str(expert_opinion)}


# @app.route("/prediction/<datapoint_id>", methods=["GET"])
# def get_prediction(datapoint_id):
#     # TODO: Only works if get_datapoint was called before
#     prediction = manager.get_current_prediction()
#     return {"result": str(prediction)}

if __name__ == "__main__":
    app.run(debug=False, port=4455, host='0.0.0.0')

import json

from flask import Blueprint, request

from service.response import response
from service.openai import openai_service

juniorbot_module = Blueprint("juniorbot_module", __name__)

@juniorbot_module.route("/task", methods = ["POST"])
def do_task():
    # simulate pair programming
    # first ask what should be done in the task.
    # then only do the task.
    # provide some bugs on the reponse code.
    data = json.loads(request.data)
    task = data["task"]
    task_response = openai_service.callopenai("juniorbot", task)
    body = {
        "response": task_response,
        "msg": "Task response obtained."
    }
    return response(200, body)

@juniorbot_module.route("/query", methods = ["POST"])
def pair_program_task():
    # get feedbacks from the human programmer as input.
    # respond to the task as per the feedback.
    # the session should be connected to the previous chats.
    data = json.loads(request.data)
    query = data["query"]
    query_response = openai_service.callopenai("juniorbot", query)
    body = {
        "response": query_response,
        "msg": "Query response obtained."
    }
    return response(200, body)

# Simulating pair programming experience.
# If i am working on a coding program or a project.
# I will find one or two team members and sit and screen share and go through and hey will do it this way or that way.
# This is what we are going to replicate using junior bot. 
# What if the bot gives the correct code at the first time sometimes.

# two days meeting in a week.
# monday ai meeting 11 am.
# wednesday discussions will be sent on the slack.

# Next week trailblazers challenge with prize for the projects going on.
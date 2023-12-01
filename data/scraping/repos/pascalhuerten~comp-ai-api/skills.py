from flask import current_app
from flask import Blueprint, jsonify, request
from ..models.chatsearcher import chatsearcher
from ..models.skillfit_model_trainer import skillfit_model_trainer
from openai import AuthenticationError
import requests

skills_blueprint = Blueprint('skills', __name__)

@skills_blueprint.route("/chatsearch", methods=["POST"])
def chatsearch():
    try:
        predictor = chatsearcher(current_app.config['EMBEDDING'], current_app.config['SKILLFIT_MODEL'], current_app.config['SKILLDBS'], request.get_json())
    except ValueError as e:
        return jsonify({"status": 400, "message": str(e)}), 400
    
    try:
        searchterms, results = predictor.predict()
    except requests.Timeout:
        return jsonify({"status": 408, "message": "Request timed out."}), 408
    except AuthenticationError:
        return jsonify({"status": 401, "message": "Invalid OpenAI API key."}), 401
    return jsonify({"searchterms": searchterms, "results": results}), 200


@skills_blueprint.route("/updateCourseSkills", methods=["POST"])
def update_course_skills():
    trainer = skillfit_model_trainer()

    data = request.get_json()
    for item in data:
        doc = None
        if "doc" in item:
            doc = item["doc"]
        else:
            return jsonify({"status": 400, "message": "Missing doc value."}), 400
        
        validationResults = []
        if "validationResults" in item:
            validationResults = item["validationResults"]
        else:
            return jsonify({"status": 400, "message": "Missing validationResults value."}), 400

        trainer.addTrainingData(doc, validationResults)
    
    return jsonify({"status": 200, "message": "Training data added."}), 200


@skills_blueprint.route("/getCourseSkills", methods=["GET"])
def get_course_skills():
    trainer = skillfit_model_trainer()
    return jsonify(trainer.getCourseSkills()), 200


@skills_blueprint.route("/trainSkillfit", methods=["GET"])
def train_skillfit():
    trainer = skillfit_model_trainer()
    training_stats = trainer.train()
    return jsonify(training_stats)


@skills_blueprint.route("/getSkillfitReport", methods=["GET"])
def report_skillfit():
    trainer = skillfit_model_trainer()
    report = trainer.getReport()
    return jsonify(report)
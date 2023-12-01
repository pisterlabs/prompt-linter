from flask import Blueprint, jsonify, request
from tree_of_thoughts.openaiModels import OpenAILanguageModel
from tree_of_thoughts.treeofthoughts import MonteCarloTreeofThoughts
from config import openai_api_key as openai_api_key
from config import chat_model as chat_model
from flask_login import login_required


solve = Blueprint("solve", __name__)


@solve.route("/solve", methods=["POST"])
@login_required
def solve_function():
    title = request.form.get("section-title")
    prompt_str = request.form.get("section-prompt")
    output_dict = {}

    model = OpenAILanguageModel(api_model=chat_model, api_key=openai_api_key)

    tree_of_thoughts = MonteCarloTreeofThoughts(model)

    input_problem = prompt_str

    num_thoughts = 5
    max_steps = 3
    max_states = 5
    pruning_threshold = 0.5
    solution = tree_of_thoughts.solve(
        input_problem,
        num_thoughts=num_thoughts,
        max_steps=max_steps,
        max_states=max_states,
        pruning_threshold=pruning_threshold,
    )
    solution_str = " ".join(solution)
    print(solution_str)
    title, result = title, {
        "question": prompt_str,
        "response": solution_str,
    }
    output_dict[title] = result
    print(output_dict)
    return jsonify(output_dict), 200

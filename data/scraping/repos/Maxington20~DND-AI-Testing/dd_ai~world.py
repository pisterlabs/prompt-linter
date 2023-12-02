from flask import Blueprint, jsonify
import openai

world_blueprint = Blueprint("world", __name__)

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()


@world_blueprint.route('/setup_world', methods=['POST'])
def setup_world_endpoint():
    prompt = "Create a D&D 5e world with a main location, a culture, and a significant event."
    world_description = generate_response(prompt)
    return jsonify({"world_description": world_description})
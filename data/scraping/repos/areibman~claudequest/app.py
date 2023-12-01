from flask import Flask, jsonify, request
from flask_cors import CORS
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

anthropic = Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="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",
)


@app.route('/api/completion', methods=['POST'])
def get_completion():
    print('!!!')
    data = request.get_json()  # Get the data sent in the POST request
    text = data.get('text', '')  # Get the 'text' field from the data

    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} {text} {AI_PROMPT}",
    )
    return jsonify(completion.completion)


if __name__ == '__main__':
    app.run(debug=True)

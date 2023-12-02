from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, g, make_response, Response
from services.claude.claude_api import AnthropicAPI
from services.claude.claude_prompt_tempalte import PromptTemplate
from services.salesforce.salesforce_api_client import SalesforceAPIClient
from flask_cors import CORS

print("Initializing HTTP Client")
app = Flask(__name__)
CORS(app, origins=['http://localhost:5173'])


@app.route('/')
def index():
    print("- Home")
    return render_template('./index.html', methods=['GET'])

@app.route('/claude', methods=['POST'])
async def claude_route_handler():
    """
    This function handles HTTP POST requests to the URL '/claude'. It generates
    text using the Anthropic API and returns the generated text as an HTTP response.

    Returns:
    An HTTP response containing the generated text
    """
    prompt = request.json.get('prompt')
    prompt = PromptTemplate.prompt_tempalte(prompt)
    print("- Claude API end point")
    # Check if the 'prompt' key is present in the **kwargs dictionary
    if prompt is None:
        return 'Error: missing "prompt" parameter', 400

    print("- Sending package")
    # Generate and return text using the Anthropic API
    try:
        async with await AnthropicAPI().generate_text(
                prompt=prompt,
                max_tokens_to_sample=3000
        ) as response:
            response_text = await response.text()
        print("- Response: ", response_text)
        return jsonify(response_text)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to generate text"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)

from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

openai.api_type = "azure"
openai.api_base = "https://careerhackers-ai.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "6fc9e8f7aa4d4bfea49dc967ade39736"

@app.route('/api/generate', methods=['POST'])
def generate():
    # Fetch prompt from request body2
    request_data = request.get_json()
    prompt = request_data.get('prompt')

    response = openai.Completion.create(
      engine="Test",
      prompt=prompt,
      temperature=0.65,
      max_tokens=800,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None
    )

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

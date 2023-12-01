from flask import Flask, request, jsonify
import openai
import os

# Set up the OpenAI API key
openai.api_key = os.environ.get("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")

# Set up the Flask app
app = Flask(__name__)

# Define the chatbot route
@app.route("/chatbot", methods=["POST"])
def chatbot():
    # Get the message from the request
    message = request.json["message"]

    # Call the OpenAI API to generate a response
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=message,
        temperature=0.5,
        max_tokens=1024,
        n=1,
        stop=None,
    )

    # Get the response text from the OpenAI API response
    response_text = response.choices[0].text.strip()

    # Return the response as JSON
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True, host="192.168.35.27")
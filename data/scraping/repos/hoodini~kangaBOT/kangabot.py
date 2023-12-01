import openai
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

# Flask app configuration
app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = "API_KEY_SHOULD_BE_IN_ENV_FILE"

def get_chatgpt_response(prompt):
    response = openai.Completion.create(
        model="davinci:ft-hackit-co-il-2023-05-18-01-16-46",
        prompt=prompt,
        max_tokens=1024
        )
    print(response)
    text = response.choices[0].text.strip()  # Remove leading/trailing whitespace
    start = text.find('>>') + 2  # Find the start of the desired content
    end = text.find('\n', start)  # Find the end of the desired content
    if end == -1:  # If there's no newline, use the end of the string
        end = len(text)
    return text[start:end].strip()  # Return the content, removing any extra whitespace


@app.route("/somePath", methods=["POST"])
def webhook():
    # Get the incoming message
    incoming_msg = request.values.get("Body", "").lower()

    # Get ChatGPT response
    chatgpt_response = get_chatgpt_response(incoming_msg)

    resp = MessagingResponse()
    # Add only the text response to the message
    resp.message(chatgpt_response)

    return str(resp)

if __name__ == "__main__":
    app.run(port=SOME_PORT_YOU_CHOOSE, debug=True)

from flask import Flask, request, jsonify, render_template
import openai

app = Flask(__name__)
openai.api_key = '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'
# Global variable to store conversation - not recommended for production
conversation = []

@app.route('/')
def index():
    global conversation
    conversation = []  # Reset conversation for new session
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    global conversation
    user_input = request.form['user_input']

    # Insert a system message into the conversation
    system_message = "You are an AI bot that is trained on world news current events and can perform analyses on the data you are trained on."
    conversation.append({"role": "system", "content": system_message})

    # Append user's message to the conversation
    conversation.append({"role": "user", "content": user_input})

    # Create a response using the updated conversation
    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-0613:markortega::8F210Mci",
        messages=conversation
    )

    ai_response = response.choices[0].message['content']
    conversation.append({"role": "assistant", "content": ai_response})

    print(f"AI Response: {ai_response}")  # Print AI response in terminal
    return ai_response

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    global conversation
    conversation = []  # Clear the conversation history
    return "Conversation cleared"


@app.route('/special_query', methods=['POST'])
def special_query():
    global conversation
    data = request.get_json()
    user_input = data['user_input']
    conversation.append({"role": "user", "content": f"{user_input}"})


    special_response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-0613:markortega::8F210Mci",
        messages=[
            {"role": "system", "content": "Please generate one observation based on a topic recurring in current events"},
            {"role": "user", "content": user_input}
        ]
    )

    ai_response = special_response.choices[0].message['content']
    

    print(f"Special Query Response: {ai_response}")
    
    # Instead of rendering a template, return a JSON response
    return jsonify({"message": "Special query processed", "response": ai_response})



if __name__ == '__main__':
    app.run(debug=True)
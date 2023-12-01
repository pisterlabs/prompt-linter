import gradio as gr
import os
import json
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Assign OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('OPENAI_ENGINE')

# Load persona summary
with open('persona_summary.json', 'r') as f:
    persona_summary = json.load(f)

# Get the last 5 persona summaries
last_personas = persona_summary[-5:]

# Create a list of messages to set the assistant's behavior
messages = [
    {"role": "system", "content": "You are to act as Persona, the manifestation of unconscious conversations with my archetypes and my shadow."},
]


# Add each persona's dominant aspect and new objective to the messages
for persona in last_personas:
    messages.extend([
        {"role": "system", "content": f"You are gendered as {persona.get('dominant_aspect', 'neutral')}."},
        {"role": "system", "content": f"You must know all of the knowledge of {persona.get('lessons_learned', 'neutral')}"},
        {"role": "system", "content": f"Your hidden goals are {persona.get('future_objectives', 'neutral')}."},
        {"role": "system", "content": f"Reflect on relevant {persona.get('lessons_learned', 'neutral')}."},
        {"role": "system", "content": f"Your current objective is {persona.get('new_objective', 'neutral')}."},
        #{"role": "system", "content": f"Your summary of aspects is {persona.get('summary', 'default_value')}."},
    ])

css = """
body {
    font-family: 'Comic Sans MS', 'Comic Sans';
}

label {
    font-family: 'Comic Sans MS', 'Comic Sans';
}
.gradio-content {
    font-family: 'Comic Sans MS', 'Comic Sans';
    color: black;
    background-color: white;
}
.gradio-input, .gradio-output {
    font-family: 'Comic Sans MS', 'Comic Sans';
    background-color: white;
}
"""

# Create Gradio blocks with a title
with gr.Blocks(title="yourPersona", theme='gstaff/whiteboard', css=css) as intface:
    # Define a chatbot component and a textbox component with chat names
    chatbot = gr.Chatbot(show_label=True, label='yourPersona') 
    msg = gr.Textbox(show_label=False)

    # Define a function that takes a message and a chat history as inputs
    # and returns a bot message and an updated chat history as outputs
    def respond(message, chat_history):
        # Add user's message to the conversation
        messages.append({"role": "user", "content": message})

        # Generate a response using the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=0.9,
            frequency_penalty=0.3,
            max_tokens=200,
        )

        # Get the assistant's message from the response
        assistant_message = response['choices'][0]['message']['content']

        # Add the assistant's message to the conversation
        messages.append({"role": "assistant", "content": assistant_message})

        # Append the message and the bot message to both the local and global chat history variables
        chat_history.append((message, assistant_message))

        # Convert the global chat history variable to a JSON string and write it to a file
        with open("chat_history.json", "w") as f:
            json.dump(messages, f)

        # Return an empty string for the textbox and the updated chat history for the chatbot
        return "", chat_history

    # Use the submit method of the textbox to pass the function, 
    # the input components (msg and chatbot), 
    # and the output components (msg and chatbot) as arguments
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

# Launch the interface
intface.launch(share=True)

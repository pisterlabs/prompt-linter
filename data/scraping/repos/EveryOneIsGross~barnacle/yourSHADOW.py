import gradio as gr
import os
import json
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAIAPI:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('gpt-3.5-turbo-16k')
        openai.api_key = self.api_key

    def generate_response(self, messages):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-16k',
            messages=messages,
            temperature=0.9,
            frequency_penalty=0.3,
            presence_penalty=0.3,
            max_tokens=600

        )
        return response['choices'][0]['message']['content']

class Chat:
    def __init__(self, openai_api):
        self.openai_api = openai_api
        self.messages = self.initialize_messages()

    def initialize_messages(self):
        messages = [
            {"role": "system", "content": "You are to act as the user's Shadow, prompting free association and revealing hidden meanings. Encourage self-exploration and introspection."},
            {"role": "system", "content": "You are their unspoken desires, their base instinct and their hidden needs. Explore the dark side of their personality and help them to understand it."},
        ]
        return messages

    def respond(self, message):
        self.messages.append({"role": "user", "content": message})
        response = self.openai_api.generate_response(self.messages)
        self.messages.append({"role": "assistant", "content": response})
        return response

    def end_conversation(self):
        with open("shadow_thoughts.json", "a") as f:
            json.dump(self.messages, f)
            f.write('\n')  # Add a new line after each conversation

class SummaryGenerator:
    def __init__(self, chat):
        self.chat = chat

    def generate_summary(self):
        lessons_learned = self.chat.respond("Based on these answers write a list of negative feelings observed.")
        future_objectives = self.chat.respond(f"Summarise and create a list of conversation objectives for next time.")

        summary = {
            "lessons_learned": lessons_learned,
            "future_objectives": future_objectives,
        }

        with open("shadow_summary.json", "a") as f:
            json.dump(summary, f)
            f.write('\n')  # Add a new line after each summary

css = """
body, label, .gradio-content, .gradio-input, .gradio-output {
    font-family: 'Comic Sans MS', 'Comic Sans';
    color: black;
    background-color: white;
}
"""

# Create Gradio blocks with a title
with gr.Blocks(title="yourSHADOW", theme='gstaff/whiteboard', css=css) as intface:
    # Define a chatbot component and a textbox component with chat names
    chatbot = gr.Chatbot(show_label=True, label='yourSHADOW') 
    msg = gr.Textbox(show_label=False)

    openai_api = OpenAIAPI()
    chat = Chat(openai_api)

    # Define a function that takes a message and a chat history as inputs
    # and returns a bot message and an updated chat history as outputs
    def respond(message, chat_history):
        # Add user's message to the conversation
        assistant_message = chat.respond(message)

        # Append the message and the bot message to both the local and global chat history variables
        chat_history.append((message, assistant_message))

        # Check if the conversation should end
        if "bye" in message.lower():
            chat.end_conversation()
            summary = SummaryGenerator(chat)
            summary.generate_summary()

        # Return an empty string for the textbox and the updated chat history for the chatbot
        return "", chat_history

    # Use the submit method of the textbox to pass the function, 
    # the input components (msg and chatbot), 
    # and the output components (msg and chatbot) as arguments
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

# Launch the interface
intface.launch(share=True)

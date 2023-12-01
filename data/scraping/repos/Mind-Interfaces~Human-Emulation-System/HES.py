# Human Emulation System (HES.py)

from gradio import Interface
from gradio.components import Textbox
import logging
import openai
import os


class HumanEmulationSystem:
    def __init__(self):
        # Define configuration settings.
        self.openai_model = "gpt-3.5-turbo"
        self.DEBUG = False  # Set to True to show API calls

        # Configure logging.
        logging.basicConfig(level=logging.DEBUG if self.DEBUG else logging.INFO)
        self.chat_history = ""

        # Read the OpenAI API key from environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key is None:
            raise EnvironmentError("OPENAI_API_KEY is not set in the environment variables.")
        openai.api_key = self.openai_api_key

        # Set cognitive contexts.
        self.context_left = "Analytic Logic, Data-Driven Thinking, Focusing on Facts and Evidence"
        self.context_right = "Creative Reasoning, Intuition, Symbolic Linking, Exploring Possibilities"
        self.context_mid = "Polymath, Seamless Viewpoint Integration, Focused on Essential Aspects"

        # Set moderator modifiers.
        self.format_mod = " (Keep your response short, on topic, well balanced and concise.) "
        self.negative_mod = "[DO NOT IDENTIFY AS an LLM, AI, language model, or AI language model]"
        self.modifiers = self.format_mod + self.negative_mod

    @staticmethod
    def chat_log(chat, prompt, mid_result):
        log = f"{chat}User(Input): {prompt}\nSystem(Output): {mid_result}\n"
        return log

    def log_debug(self, message):
        # Send debug output to console.
        if self.DEBUG:
            logging.debug(message)

    def call_left_hemisphere(self, prompt, left_lobe):
        # Generate an analytical response.
        request_params = {
            "model": self.openai_model,
            "messages": [{"role": "system", "content": left_lobe},
                         {"role": "user", "content": prompt}],
            "max_tokens": 250,
        }
        self.log_debug(f"Left Hemisphere Request: {request_params}")
        response = openai.ChatCompletion.create(**request_params)
        self.log_debug(f"Left Hemisphere Response: {response}")
        return response.choices[0].message['content']

    def call_right_hemisphere(self, prompt, right_lobe):
        # Generate a creative response.
        request_params = {
            "model": self.openai_model,
            "messages": [{"role": "system", "content": right_lobe},
                         {"role": "user", "content": prompt}],
            "max_tokens": 250,
        }
        self.log_debug(f"Right Hemisphere Request: {request_params}")
        response = openai.ChatCompletion.create(**request_params)
        self.log_debug(f"Right Hemisphere Response: {response}")
        return response.choices[0].message['content']

    def call_model(self, prompt, left_lobe, right_lobe, response_moderator):
        # Integrate multiple perspectives into a multi-dimensional response.
        left_result = self.call_left_hemisphere(prompt, left_lobe)
        right_result = self.call_right_hemisphere(prompt, right_lobe)
        # Compile responses to synthesize an integrated response.
        combined = f"{self.chat_history}\nQuery(Input): {prompt}\n"
        combined += f"[Left Hemisphere(Internal): {left_result}]\n"
        combined += f"[Right Hemisphere(Internal): {right_result}]\n"
        combined += "Response(Output):"
        # Enforce negative modifiers on the response moderator.
        moderator = response_moderator + self.modifiers
        # Generate a moderated response.
        request_params_mid = {
            "model": self.openai_model,
            "messages": [{"role": "system", "content": moderator},
                         {"role": "user", "content": combined}],
            "max_tokens": 500,
        }
        self.log_debug(f"Mid Brain Request: {request_params_mid}")
        response_mid = openai.ChatCompletion.create(**request_params_mid)
        self.log_debug(f"Mid Brain Response: {response_mid}")
        # Compile conversation for chat log and display Response.
        mid_result = response_mid.choices[0].message['content']
        self.chat_history = self.chat_log(self.chat_history, prompt, mid_result)
        return self.chat_history, left_result, right_result, mid_result


# Create an instance of the Human Emulation System
HES = HumanEmulationSystem()

# Gradio Web GUI
GUI = Interface(
    HES.call_model,
    inputs=[
        Textbox(lines=2, placeholder="Enter your query here...", label="Input Prompt"),
        Textbox(lines=1, value=HES.context_left, label="Analytic Logic"),
        Textbox(lines=1, value=HES.context_right, label="Creative Reasoning"),
        Textbox(lines=1, value=HES.context_mid, label="Response Moderator"),
    ],
    outputs=[
        Textbox(lines=2, placeholder="", label="Chat Log"),
        Textbox(label="Left Hemisphere Response"),
        Textbox(label="Right Hemisphere Response"),
        Textbox(label="Synthesized Response"),
    ],
    live=False,
    title='Human Emulation System',
    description="Explore the emulation of human cognition by synthesizing logical and creative dichotomy."
)

# Initialize
GUI.launch()

# EOF // 2023 MIND INTERFACES, INC. ALL RIGHTS RESERVED.

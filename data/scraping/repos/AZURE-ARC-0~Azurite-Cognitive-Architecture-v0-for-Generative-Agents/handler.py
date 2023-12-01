from services.generation_text.openai_text import OpenAITextGeneration
from services.generation_text.prompts.messages import Messages
import base64


class DataHandler():
    def __init__(self, persona_image=None):
        self.openai_text = OpenAITextGeneration()
        self.messages = Messages()
        self.image_path = persona_image

    def handle_chat(self, user_message, role=None, num_messages=None,):
        prompt_list = self.messages.get_context(
            user_message,
            role,
            num_messages,
        )
        return self.openai_text.send_chat_complete(prompt_list)

    def handle_image(self):
        with open(self.image_path, "rb") as f:
            content = f.read()
        return base64.b64encode(content).decode("utf-8")

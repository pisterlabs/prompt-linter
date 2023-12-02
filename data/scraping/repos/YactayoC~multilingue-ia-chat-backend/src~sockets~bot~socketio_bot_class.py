from cryptography.fernet import Fernet
from flask_socketio import emit
import unicodedata
import openai

from src.services.common.message_service import MessageService


class MessageBotHandler:
    api_key = ""  #! No subirlo
    context_file = "context.txt"
    context = ""
    context_initialized = False

    def __init__(self, socketio_instance):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        self.private_rooms = {}
        self.socketio = socketio_instance
        self.message_service = MessageService()
        self.context = self.read_context_from_file(self.context_file)

    def encrypt(self, message):
        encrypted_message = self.cipher_suite.encrypt(message.encode())
        return encrypted_message.decode("ascii")

    def decrypt(self, encrypted_message):
        decrypted_message = self.cipher_suite.decrypt(encrypted_message.encode("ascii"))
        decrypted_message = decrypted_message.decode("utf-8")
        decrypted_message = unicodedata.normalize("NFKD", decrypted_message)
        return decrypted_message

    def read_context_from_file(self, file_path):
        with open(file_path, "r") as file:
            return file.read()

    def interact_with_chatgpt(self, data):
        message = data["message"]
        room_name = data["room_name"]
        message = message.lower()

        prompt = self.context + "\n" + message
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=10,
            temperature=0.7,
            api_key=self.api_key,
        )

        # Actualizar el contexto con la respuesta generada por ChatGPT
        # context = response.choices[0].text.encode("cp1252").decode()
        context = response.choices[0].text.strip()
        print(context)
        self.socketio.emit(
            "response_bot",
            {"room_name": room_name, "message": context, "user": "Bot"},
            room=room_name,
        )

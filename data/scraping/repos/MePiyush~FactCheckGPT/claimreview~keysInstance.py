import os
from dotenv import load_dotenv
import openai


class keysInstance:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAIKEY")
        self.pinecone_api_key = os.getenv("PCKEY")
        self.email = os.getenv("EMAIL")
        self.password = os.getenv("PASSWORD")
        openai.api_key = self.openai_api_key
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

    def get_openai_api_key(self):
        return self.openai_api_key

    def get_pinecone_api_key(self):
        return self.pinecone_api_key

    def get_password_api_key(self):
        return self.password

    def get_email_api_key(self):
        return self.email

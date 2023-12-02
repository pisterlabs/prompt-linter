import os
import openai
from dotenv import load_dotenv

load_dotenv()
class EnvService:
    def __init__(self):
        pass

    @staticmethod
    def get_openai_api_key():
        try:
            return os.getenv('OPENAI_API_KEY')
        except:
            raise PermissionError('Could not get OPENAI_API_KEY from environment')

    @staticmethod
    def get_openai_organization():
        try:
            return os.getenv('OPENAI_ORGANIZATION')
        except:
            raise PermissionError('Could not get OPENAI_ORGANIZATION from environment')


openai.openai_api_key = EnvService.get_openai_api_key()

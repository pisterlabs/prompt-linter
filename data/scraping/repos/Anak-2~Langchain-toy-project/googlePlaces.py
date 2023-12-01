import os
import dotenv
from langchain.tools import GooglePlacesTool

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)


def get_google_map():
    return GooglePlacesTool()

import configparser
import os
from langchain.tools import GooglePlacesTool


class CustomPlacesTool(GooglePlacesTool):
    def __init__(self, config_path='../resources/config.ini'):
        config = configparser.ConfigParser()
        config.read(config_path)
        os.environ["GPLACES_API_KEY"] = config['GOOGLE']['PLACES_KEY']
        super().__init__()

    def run(self, query: str):
        return super().run(
            query.replace('위치', '')
        )

#################################################################
#################### MONKEY HEAD PROJECT ########################
#################### Dylan L.R. Pollock #########################
############# GIZMO-V1.0-BETA | FILE: 'GIZMO.py' ################
######## Website: www.dlrp.ca | Contact: admin@dlrp.ca ##########
######## Created: [10-12-2023] | Last Modified: [12-12-2023] ####
############# Description: [GIZMO CORE PROGRAM] #################
################## License: GNU GPL V3.0 ########################
#################################################################

import os
import sys
import datetime
import json
import logging
import openai
from dotenv import load_dotenv

class CLI:
    @staticmethod
    def print_message(message, level):
        print(f"{message} ({level})")
        print(f"{level}: {message}")

    @staticmethod
    def handle_exception(exception, context):
        print(f"Error in {context}: {exception}")
    try:
        # Code that might raise an exception
    except Exception as exception):
        logging.exception(f"Error in {context}: {exception}")
    


class CoreGizmo:
    def __init__(self, config, memory, cli):
        self.cli = cli
        self.default_config = config
        self.memory = memory
        self.setup_openai_api()
        self.load_config_and_memory()
        self.setup_environment()
        self.initial_message()

    def setup_openai_api(self):
        try:
            load_dotenv()
            self.api_key = os.getenv('API_KEY')
            if not self.api_key:
                raise ValueError("API key not found in environment variables")
            openai.api_key = self.api_key
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            logging.basicConfig(filename=f'LOG_{timestamp}_GIZMO.json', level=logging.DEBUG)
        except Exception as e:
            self.cli.handle_exception(e, "setup_openai_api")
            sys.exit(1)

    def setup_environment(self):
        try:
            use_custom_path = self.default_config.get("use_custom_path", False)
            custom_path = self.default_config.get("custom_path", "") if use_custom_path else ""
            self.ensure_folders_exist(custom_path)
        except Exception as e:
            self.cli.handle_exception(e, "setup_environment")

    def ensure_folders_exist(self, path):
        if path and not os.path.exists(path):
            os.makedirs(path)
        if path and not os.path.isdir(path):
            raise NotADirectoryError(f"Path {path} is not a directory")

    def load_config_and_memory(self):
        try:
            self.config = self.load_config()
            self.memory = self.load_memory()
        except Exception as e:
            self.cli.handle_exception(e, "load_config_and_memory")

    def load_config(self):
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.default_config

    def load_memory(self):
        try:
            with open('MEMORY.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def initial_message(self):
        program_name = os.path.basename(__file__).replace('.py', '')
        formatted_program_name = ' '.join(word.capitalize() for word in program_name.split('_'))
        self.cli.print_message(f"Starting '{formatted_program_name}.py'... {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "INFO")

    def main(self):
        # Main application logic
        self.cli.print_message("Running main application logic", "INFO")
        # TODO: Implement your application's main logic here

    def graceful_exit(self):
        self.cli.print_message("Exiting Gizmo application.", "INFO")
        sys.exit()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    cli = CLI()

    default_config = {
        "use_custom_path": False,
        "custom_path": "./data"
    }
    default_memory = []

    try:
        gizmo = CoreGizmo(default_config, default_memory, cli)
        gizmo.main()
    except Exception as e:
        cli.handle_exception(e, "Main Application")
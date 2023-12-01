import sys
import os
import openai
import requests
import json
import argparse
import tempfile
import time
import shutil
from datetime import datetime
from dotenv import load_dotenv
from importlib.machinery import SourceFileLoader

# Load environment variables from .env file
load_dotenv()

class DescriptionProcessor:
    def __init__(self, url, theme, engine, max_tokens=200, retries=3, retry_delay=60, max_requests=1000, interval=60):
        self.url = url
        self.theme = theme
        self.folder = theme.replace(" ", "_")   # Replaced spaces with underscores to avoid issues with folder paths
        self.engine = engine
        self.max_tokens = max_tokens
        self.retries = retries
        self.retry_delay = retry_delay
        self.max_requests = max_requests
        self.interval = interval
        self.request_counter = 0
        self.start_time = time.time()
        self.file_types = [
            "trait_descriptions",
            "career_descriptions",
            "zone_descriptions",
            "sim_descriptions",
            "interaction_descriptions",
        ]
        # Ensure the theme-named sub-directory exists, if not create it
        if not os.path.exists(self.folder):
           os.mkdir(self.folder)

    def process_sim_descriptions(self, descriptions):
        """
        Process sim descriptions by generating new descriptions with the given theme.

        Args:
            descriptions (List[str]): A list of sim descriptions.

        Returns:
            List[str]: A list of new themed descriptions.
        """
        new_descriptions = []
        for description in descriptions:
            prompt = f"Recreate the following description with a {self.theme} theme:\n{description}\n"
            new_description = self.generate_new_data(prompt)
            new_descriptions.append(new_description)
            
        return new_descriptions

    def process_career_descriptions(self, data):
        """
        Process career descriptions by generating new descriptions with the given theme.

        Args:
            data (dict): The original career descriptions.

        Returns:
            dict: The processed career descriptions.
        """
        new_data = {}

        # print(f"DEBUG process_career_descriptions: {data}")

        for career_category, career_category_info in data.items():
            new_career_category = {}
            for career_id, career_info in career_category_info.items():
                new_career_info = {}
                for property_name, description in career_info.items():
                    if property_name == 'name':
                        prompt_title = 'name'
                    elif property_name == 'description':
                        prompt_title = 'description (do not rewrite content inside brackets or braces)'
                    elif property_name == 'sentient_sims_description':
                        prompt_title = 'brief description (do not rewrite content inside brackets or braces)'
                    else:
                        continue   # skip if property_name is not one we want to recreate
                        
                    prompt = f"Recreate the following {prompt_title} with a {self.theme} theme:\n{description}\n"
                    new_description = self.generate_new_data(prompt)
                    new_career_info[property_name] = new_description
                new_career_category[career_id] = new_career_info
            new_data[career_category] = new_career_category

        return new_data

    def process_trait_descriptions(self, data):
        """
        Process trait descriptions by generating new descriptions with the given theme.

        Args:
            data (str): The original trait description.

        Returns:
            str: The processed trait description.
        """
        prompt = f"The following is used to describe someone (trait), please reword using a {self.theme} theme:\n{data}\n"
        new_data = self.generate_new_data(prompt)
        return new_data

    def process_zone_descriptions(self, zone_info):
        """
        Process a zone's description by generating a new description with the given theme.

        Args:
            zone_info (dict): The original zone information.

        Returns:
            dict: The processed zone information.
        """

        new_zone_info = {}
        if isinstance(zone_info, dict):
            prompt = f"Recreate the following data with a {self.theme} theme:\n"
            prompt += f" Name: {zone_info['name']}\n Type: {zone_info['type']}\n Description: {zone_info['description']}\n"
            new_description = self.generate_new_data(prompt)
            new_zone_info = {
                "name": zone_info["name"],
                "type": zone_info["type"],
                "description": new_description,
            }
        else:
            print("WARNING: Zone information was not provided as a dictionary.")
        return new_zone_info

    def process_interaction_descriptions(self, data):
        """
        Process interaction descriptions by generating new actions with the given theme.

        Args:
            data (dict): The original interaction descriptions.

        Returns:
            dict: The processed interaction descriptions.
        """

        # print(f"DEBUG process_interaction_descriptions: {data}")

        new_data = {}
        for unique_id, interactions in data.items():
            if isinstance(interactions, list):
                new_interactions = []
                for interaction in interactions:
                    prompt = f"Recreate the following action with a {self.theme} theme:\n{interaction}\n"
                    new_interaction = self.generate_new_data(prompt)
                    new_interactions.append(new_interaction)
                new_data[unique_id] = new_interactions
            else:
                print(f"Unexpected data format for unique_id {unique_id}, value is not a list but a {type(interactions)}")
        return new_data

    def get_dictionary_from_location(self, file_type):
        """
        Get dictionary from URL or local file.

        Args:
            file_type (str): The type of the file (e.g., "sim_descriptions").

        Returns:
            dict: The data from the file.
        """
        if self.url.endswith('.json') or self.url.endswith('.py'):
            location = self.url
        else:
            json_location = os.path.join(self.url, f"{file_type}.json")
            py_location = os.path.join(self.url, f"{file_type}.py")
            location = json_location if os.path.exists(json_location) else py_location

        if location.startswith('http'):
            response = requests.get(location)
            if location.endswith('.py'):
                data = self.import_variable_from_module(location, file_type)
            else:
                data = response.json()
        else:
            if location.endswith('.py'):
                data = self.import_variable_from_module(location, file_type)
            else:
                with open(location, 'r') as file:
                    data = json.load(file)

        return data

    def process_descriptions(self, process_func, data, output_file):
        """
        Process descriptions by generating new descriptions with the given theme.

        Args:
            process_func (function): The function to use for processing the descriptions.
            data (list): The original descriptions.
            output_file (str): The name of the output file.

        Returns:
            list: The processed descriptions.
        """ 
        new_data = {}
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        
        output_file = os.path.join(self.folder, output_file) 

        for key, value in data.items():
            new_data[key] = process_func(value)

            with open(temp_file.name, 'w') as f:
                json.dump(new_data, f)
            
        temp_file.close() 

        # Check if output file exists, if so rename it to avoid conflicts
        if os.path.exists(output_file):
            # Get current date and time as a string to use in the renaming process
            dt_string = datetime.now().strftime("%Y%m%d%H%M%S")
            base_name, ext = os.path.splitext(output_file)
            new_name = f"{base_name}_{dt_string}{ext}"
            os.rename(output_file, new_name)
        
        shutil.move(temp_file.name, output_file)
        return new_data

    def process_json_file(self, file_type):
        """
        Process a specific file type by generating new descriptions with the given theme.

        Args:
            file_type (str): The type of the file (e.g., "sim_descriptions").

        Returns:
            dict: The processed data.
        """
        data = self.get_dictionary_from_location(file_type)

        process_funcs = {
            "career_descriptions": self.process_career_descriptions,
            "sim_descriptions": self.process_sim_descriptions,
            "trait_descriptions": self.process_trait_descriptions,
            "zone_descriptions": self.process_zone_descriptions,
            "interaction_descriptions": self.process_interaction_descriptions,
        }
        process_func = process_funcs.get(file_type)
        if process_func is None:
            print(f"Unsupported file type: {file_type}")
            return
        output_file = f"{file_type}.json"
        new_data = self.process_descriptions(process_func, data, output_file)
        return new_data

    def import_variable_from_module(self, module_path, var_name):
        """
        Import a specified variable from a Python module.

        Args:
            module_path (str): The file path to the Python module.
            var_name (str): The name of the variable to import from the module.

        Returns:
            The imported variable from the Python module.
        """
        mod = SourceFileLoader("mod", module_path).load_module()
        var = getattr(mod, var_name)
        return var

    def process_python_file(self, file_type):
        """
        Process a specific file type by generating new descriptions with the given theme,
        convert the .py into .json and then makes use of process_json_file.

        Args:
            file_type (str): The type of the file without the extension (e.g., "sim_descriptions").

        Returns:
            dict: The processed data.
        """

        # Import the data variable from the .py file
        module_path = os.path.join(self.url, f"{file_type}.py")
        data = self.import_variable_from_module(module_path, file_type) 

        # Convert Python data to json file
        json_file = f"{file_type}.json"
        
        # Check if JSON file exists, if so rename it to avoid conflicts
        if os.path.exists(json_file):
            dt_string = datetime.now().strftime("%Y%m%d%H%M%S")  # Get current date and time as a string
            base_name, ext = os.path.splitext(json_file)
            new_name = f"{base_name}_{dt_string}{ext}"  # Rename the file by appending current date and time
            os.rename(json_file, new_name)

        # Debug: Check the Python dictionary to JSON conversion
        try:
            json_data = json.dumps(data)
            # print(f"JSON data: {json_data}")  # Output the resulting JSON
        except Exception as e:
            print(f"Error during conversion Python dict to JSON: {str(e)}")

        # TODO this is broken but only for 'zone', the json file has keys but not values.
        # Write to JSON file
        with open(json_file, 'w') as f:
            f.write(json_data)
        
        # Process the json file
        return self.process_json_file(file_type)

    def generate_new_data(self, prompt):
            """
            Generate new data using the OpenAI API.

            Args:
                prompt (str): The prompt to use for generating the new data.

            Returns:
                str: The generated text.
            """

            print("---")
            print(f"Input theme: {self.theme}")
            print(f"Input prompt: {prompt}")
            print(f"Input engine: {self.engine}")
            print(f"Input max_tokens: {self.max_tokens}")
            self.request_counter += 1
        
            for retry in range(self.retries):
                try:
                    response = openai.Completion.create(
                        engine=self.engine,
                        prompt=prompt,
                        max_tokens=self.max_tokens,
                        n=1,
                        stop=None,
                        temperature=0.8,
                    )

                    generated_text = response.choices[0].text.strip()
                    print(f"Generated text:\n{generated_text}")

                    # reset request counter and start time if interval has passed
                    if time.time() - self.start_time > self.interval:
                        self.start_time = time.time()
                        self.request_counter = 0

                    # sleep if max requests within interval has been reached
                    if self.request_counter >= self.max_requests:
                        sleep_time = self.interval - (time.time() - self.start_time)
                        print(f"Reached maximum request limit. Sleeping for {sleep_time} seconds.")
                        time.sleep(sleep_time)
                        # after sleep, reset request counter and start time
                        self.start_time = time.time()
                        self.request_counter = 0

                    return generated_text
                except Exception as e:
                    if retry < self.retries - 1:
                        print(f"Error: {e}. Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                    else:
                        print(f"Error: {e}. Maximum retries reached.")
                        raise

    def process_all(self):
        if not self.url.endswith('.py'):
            for file_type in self.file_types:
                print(f"Processing python file {file_type}")
                self.process_python_file(file_type)
        else:
            file_type = os.path.splitext(os.path.basename(self.url))[0]
            print(f"Processing json file {file_type}")
            self.process_json_file(file_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data files")
    parser.add_argument("-u", "--url", default=".",
                        help="URL or path to the .py file or directory containing the .py files")
    parser.add_argument("-t", "--theme", default="whimsical and playful",
                        help="Theme for processing data files")
    parser.add_argument("-e", "--engine", default="text-davinci-002",
                        help="OpenAI engine to be used for text generation")
    parser.add_argument("-m", "--max_tokens", type=int, default=200,
                        help="Maximum number of tokens to generate")
    parser.add_argument("-p", "--prompt", default="",
                        help="Custom prompt to be used in text generation")
    parser.add_argument("-s", "--settings", default="settings.json",
                        help="Path to the settings JSON file containing the OpenAI API key")
    args = parser.parse_args()

    if not openai.api_key:
        openai.api_key = get_openai_api_key(args.settings)

    processor = DescriptionProcessor(args.url, args.theme, args.engine)
    processor.process_all()


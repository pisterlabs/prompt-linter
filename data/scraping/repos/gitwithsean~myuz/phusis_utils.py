import re, time
from termcolor import colored
from django.apps import apps
from functools import wraps
from typing import Callable
import random

def extract_backtick_enclosed_content(text):
    pattern = r"```(.*?)```"
    match = re.search(pattern, text, re.DOTALL)  # re.DOTALL makes . match also the newline character
    if match:
        return match.group(1)  # return the content enclosed by backticks
    else:
        return None  # return None if no match is found

def camel_case_to_underscore(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def camel_case_to_spaced(name):
    # insert space before every capital letter
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    # convert to lowercase
    name = name.lower()
    return name


def spaced_to_underscore(name):
    return name.replace(" ", "_")


def is_valid_init_json(json_data):
    if not 'class_name' in json_data or not 'properties' in json_data or not 'name' in json_data['properties']:
        return False
    else:
        return True


def dict_to_md(dictionary, indent=0):
    md_string = ""
    for key, value in dictionary.items():
        md_string += '  ' * indent + f"- **{key}**: "
        if isinstance(value, dict):
            md_string += "\n" + dict_to_md(value, indent + 1)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    md_string += "\n" + dict_to_md(item, indent + 1)
                else:
                    md_string += f"{item}, "
            md_string = md_string.rstrip(", ") + "\n"
        else:
            md_string += f"{value}\n"
    return md_string


def memorize_chat(prompt, response, responder):
    from phusis.agent_memory import ProjectMemory
    new_chat_log = responder.add_to_chat_logs(prompt, response)
    ProjectMemory().add_chat_log_db_instance_to_pinecone_memory(new_chat_log)


def memorize_chat(chat_log):
    from phusis.agent_memory import ProjectMemory
    # print(colored(f"agent_uitls.memorize_chat(): {chat_log}", "yellow"))
    ProjectMemory().add_chat_log_db_instance_to_pinecone_memory(chat_log)    


def get_embeddings_for(text):
    from phusis.apis import OpenAiAPI
    return OpenAiAPI().get_embeddings_for(text)


def get_user_agent_singleton():
    from phusis.agent_models import UserAgentSingleton
    return UserAgentSingleton()


def load_model_and_return_instance_from(json_data, app_name):
    """
    With dict json_data and app_name and parms either create or find an instance based on the provided JSON data in app_name and return it.
    
    Args:
        json_data (dict): A dictionary containing the JSON data for the model.
        app_name (str): The name of the app to search for or create the model in.
    
    Returns:
        object: An instance of the model created or updated based on the JSON data.
    """
    
    new_project_obj = {}
    expected_json = {
        "class_name": "ModelClassName",
        "properties": {
            "name": "Instance Name"
        }
    }
 
    if is_valid_init_json(json_data):
        try:
            model_class = apps.get_model(app_name, f"{json_data['class_name']}")
        except LookupError:
            print(colored(f"agent_utils.create_project_model_from_instance: class_name {json_data['class_name']} not found in globals()", "red"))

        # print(colored(f"agent_utils.create_project_model_from_instance: model_class {model_class}. json_data['properties']['name'] {json_data['properties']['name']}", "green"))

        new_project_obj, created = model_class.objects.update_or_create(name=json_data['properties']['name'])

        new_project_obj.set_data(json_data['properties'])
        new_project_obj.save()
        s = f"found and updated with:\n{json_data['properties']}"
        if created: s = "created"
        # print(colored(f"agent_utils.create_project_model_from_instance: {new_project_obj.name} {s}", "green"))
        
    else:
        print(colored(f"agent_utils.create_project_model_from_instance: JSON data for model not valid, minimum expected schema below","red"))
        print(colored(f"Data received: {json_data}", "red"))
        print(colored(f"Minimum expected: {expected_json}", "yellow"))

    return new_project_obj


def find_and_update_or_create_attribute_by(attr_name, model_class):
    """
    Args:
        attr_name (str): The name of the attribute to find or create.
        model_class (class): The model class to search for the attribute in.

    Returns:
        tuple: A tuple containing the attribute class and the attribute instance.
    """

    # print(colored(f"agent_utils.find_and_update_or_create_attribute_by(): attr_name: {attr_name}\nmodel_class : {model_class}", "yellow"))
    
    # Check if the attribute exists in the model class
    if hasattr(model_class, attr_name):
        return model_class, getattr(model_class, attr_name)

    # print(colored(f"model class : {model_class}", "yellow"))

    # Check if the attribute exists in any related models
    for related_object in model_class._meta.related_objects:
        related_model_class = related_object.related_model

        if hasattr(related_model_class, attr_name):
            # Create an instance of the related model to return
            instance = related_model_class.objects.get(**{related_object.field.name: model_class})
            return related_model_class, instance
        else:
            #If the Attribute wasn't found, create it 
            new_attribute, created = model_class.objects.update_or_create(name=attr_name)
            return related_model_class, new_attribute


def retry_with_exponential_backoff(
    max_retries: int = 5, initial_delay: float = 0.1
) -> Callable:
    def decorator_retry(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay

            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if retries == max_retries:
                        raise
                    sleep_time = delay * (2 ** retries)
                    jitter = sleep_time * 0.1 * random.random()
                    sleep_time += jitter
                    print(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    retries += 1

        return wrapper

    return decorator_retry


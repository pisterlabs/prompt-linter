import yaml
import openai

def load_systems():
    with open('systems.yaml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)['systems']

def load_characters():
    with open('characters.yaml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)['characters']

def load_models():
    with open('models.yaml', 'r', encoding='utf-8') as file:
        models_config = yaml.safe_load(file)
        # Normalize the model names
        for key in models_config['openai_models']:
            models_config['openai_models'][key]['name'] = key
        return models_config

def setup_openai(api_key):
    openai.api_key = api_key
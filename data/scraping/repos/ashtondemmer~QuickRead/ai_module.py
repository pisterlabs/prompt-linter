import cohere
from CTkMessagebox import CTkMessagebox
import os


def run_ai(input, length, root):
    length = length.split()[0].lower()

    # Construct the path using the scripts dir
    script_directory = os.path.dirname(os.path.abspath(__file__))
    api_key_path = os.path.join(script_directory, 'api_key.env')
    
    with open(api_key_path, 'r') as env_file:
        for line in env_file:
            key, value = line.strip().split('=')
            os.environ[key] = value
    co = cohere.Client(os.environ['COHERE_API_KEY'])
    
    try:
        response = co.summarize(
            text=input,
            model='command',
            length=length,
            extractiveness='medium'
        )
        return response.summary 
    except:
        CTkMessagebox(master=root, title="Error", message="You must submit at least 250 characters and can only make 5 summaries/min. ", icon="cancel", width=400, height=50, button_width=180, button_height=100, topmost=False)

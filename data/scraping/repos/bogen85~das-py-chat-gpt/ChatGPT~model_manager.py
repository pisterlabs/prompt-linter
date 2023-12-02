import requests

def get_model_list(api_key):
    # Get list of available models from OpenAI API
    response = requests.get(
        "https://api.openai.com/v1/models",
        headers={
            "Authorization": f"Bearer {api_key}"
        }
    )
    models = response.json()['data']
    return models

def choose_model(models):
    # Print list of models and ask user to choose one
    print("Available models:")
    #pprint(models)
    for i, model in enumerate(models):
        print(f"{i+1}: {model['id']}")
    choice = int(input("Choose a model by its number: "))
    return models[choice-1]

def save_model(model, filepath):
    # Save chosen model to file
    with open(filepath, 'w') as f:
        f.write(model['id'])

def load_model(filepath):
    # Load model from file
    with open(filepath, 'r') as f:
        model_id = f.read()
    return model_id

def get_model(api_key, filepath):
    # Try to load model from file, if it doesn't exist, ask user to choose one and save it to file
    try:
        model_id = load_model(filepath)
    except FileNotFoundError:
        models = get_model_list(api_key)
        model = choose_model(models)
        save_model(model, filepath)
        model_id = model['id']
    return model_id

# CudaText: lexer_file="Python"; tab_size=4; tab_spaces=Yes; newline=LF;

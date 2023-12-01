import cohere
from google.cloud import vision
from google.oauth2 import service_account
from taipy.gui import Gui, notify
import keys

def initialize_vision_client(service_account_key_path):
    """Initialize the Vision client."""
    credentials = service_account.Credentials.from_service_account_file(service_account_key_path)
    client = vision.ImageAnnotatorClient(credentials=credentials)
    return client

def detect_text_local(path, client):
    """Detects text in the file."""
    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("Texts:")

    # Optimize this part
    i = 0
    for text in texts:
        if i == 1:
            break
        #print(f'\n"{text.description}"')
        result = text.description
        return result
        i += 1

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

def detect_text_uri(uri, client):
    """Detects text in the file located in Google Cloud Storage or on the Web."""
    image = vision.Image()
    image.source.image_uri = uri

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("Texts:")
    
    # Optimize this part
    i = 0
    for text in texts:
        if i == 1:
            break
        #print(f'\n"{text.description}"')
        result = text.description
        return result
        i += 1

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

def summarize_text(text):
    # Initialize the Cohere client with your API key
    co = cohere.Client(keys.cohere_key)  # This is your trial API key

    try:
        response = co.summarize(
            text=text,
            length='auto',
            format='auto',
            model='command',
            additional_command='',
            temperature=0.3,
        )
    except cohere.error.CohereAPIError as e:
        error_message = "Error: invalid request - text must be longer than 250 characters"
        print(error_message)
        return error_message

    # Print the summarized text
    return response.summary


# Initialize the Vision client
service_account_key_path = 'daring-spirit-404120-e0243e440d8a.json'
vision_client = initialize_vision_client(service_account_key_path)

content = ""
inputText = "URI"
ScannedText = "Nothing Here!"
SummarizedText = "Nothing Here!"

stylekit = {
  "color_primary": "#BADA55",
  "color_secondary": "#C0FFE",
}

# Definition of the page
page = """
<|container|
# SummarizeMate

<|Enter URI|> \n\n
<|{inputText}|input|>
<|{content}|file_selector|on_action=process_upload|extensions= .jpg, .png|>
<|Convert to text|button|on_action=on_button_action|>

####Scanned Text\n 
<|card|
<|{ScannedText}|>
|>
####Summarized Text\n 
<|card|
<|{SummarizedText}|>
|>
|>
"""

def process_upload(state):
    state_variable = state.content
    print(state.content)
    return state_variable

def on_button_action(state):
    if (state.inputText == "URI"):
        filePath = process_upload(state)
        state.ScannedText = detect_text_local(filePath, vision_client)
        state.SummarizedText = summarize_text(state.ScannedText)
    else:
        state.ScannedText = detect_text_uri(state.inputText, vision_client)
        state.SummarizedText = summarize_text(state.ScannedText)

def on_change(state, var_name, var_value):
    if var_name == "text" and var_value == "Reset":
        state.inputText = ""
        return

Gui(page).run(dark_mode=True, use_reloader=True, stylekit=stylekit)

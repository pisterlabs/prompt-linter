#!/usr/bin/env python
# coding: utf-8

# In[7]:


import speech_recognition as sr
from pydub import AudioSegment
import json
import openai
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


def one_for_all_V3():

    # Create a recognizer object
    recognizer = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Speak something...")

        # Adjust the ambient noise level (optional)
        recognizer.adjust_for_ambient_noise(source)

        # Record audio until 3 seconds of silence or a phrase is detected
        audio_data = recognizer.listen(source, timeout=3)

    try:
        # Recognize speech using the Google Speech Recognition API
        text = recognizer.recognize_google(audio_data)
        print(f"Speech recognized: {text}")
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

    # Print the recognized speech after recording
    print("Finished recording.")

    
    #generate_questions_with_answers from chatgpt API
    instruction = "in this form Q: A) B) C) D)"
    openai.api_key = 'your_key'
    # Define the instruction and prompt for the model
    prompt = f"{text}{instruction}"

    # Generate questions and answer options using the ChatGPT model
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        n=1,  # Generate three questions
        stop=None,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    #full text for the form
    full_text = response["choices"][0]["text"]
    
    print("finished generating full_text")

    #google forms API , getting the link
    SCOPES = ['https://www.googleapis.com/auth/forms.body']

    # Load client secrets
    with open('client_secrets.json') as secrets_file:
        secrets = json.load(secrets_file)

    # Create credentials flow
    flow = InstalledAppFlow.from_client_config(
        secrets,
        scopes=SCOPES
    )

    # Authenticate and authorize access
    credentials = flow.run_local_server(port=0)

    # Check if token file already exists
    token_path = 'token.json'
    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(
                secrets,
                scopes=SCOPES
            )
            credentials = flow.run_local_server(port=0)

        # Save credentials to token file
        with open(token_path, 'w') as token_file:
            token_file.write(credentials.to_json())

    # Build Google Forms service
    service = build('forms', 'v1', credentials=credentials)

    # Request body for creating a form
    NEW_FORM = {
        "info": {
            "title": "Machine learning QUIZ",
        }
    }


    # Create the form
    response = service.forms().create(body=NEW_FORM).execute()

    # Get the form ID
    form_id = response["formId"]

    # Create a list to store question requests
    question_requests = []

    questions_text = []
    answer_options_dict = {}

    question_lines = full_text.strip().split('\n\n')
    for lines in question_lines:
        lines = lines.strip().split('\n')
        question = lines[0].lstrip('Q:').strip()
        answer_options = [option.strip() for option in lines[1:] if option.strip()]
        questions_text.append(question)
        answer_options_dict[question] = answer_options


    # Create the questions in the form
    for question, answer_options in answer_options_dict.items():
        question_body = {
            "createItem": {
                "item": {
                    "title": question,
                    "questionItem": {
                        "question": {
                            "required": True,
                            "choiceQuestion": {
                                "type": "RADIO",
                                "options": [{"value": option} for option in answer_options],
                                "shuffle": True
                            }
                        }
                    },
                },
                "location": {
                    "index": 0
                }
            }
        }
        question_requests.append(question_body)



    # Add the questions to the form
    request_body = {"requests": question_requests}
    question_setting = service.forms().batchUpdate(formId=form_id, body=request_body).execute()

    # Construct the form URL
    form_url = f"https://docs.google.com/forms/d/{form_id}/viewform"
    return text,form_url



# In[ ]:





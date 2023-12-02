import io
import json
import time
from openai import OpenAI
import sys
import requests
import os
if os.path.dirname(os.path.realpath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import keys
import apiDataHandler
    
def askGPT(emailText, files, imageInfo=[]):
    client = OpenAI(api_key=keys.openAI_APIKEY)

    for index, file_ in enumerate(files):
        files[index] = client.files.create(
        file=file_,
        purpose='assistants'
        ).id

    # Add the file to the assistant
    textFileAssistant = client.beta.assistants.create(
    instructions="You are a helpful robot.",
    model="gpt-4-1106-preview",
    tools=[{"type": "retrieval"}],
    file_ids=[]
    )

    if len(files) > 0:
        content_text = "Extract ALL flight details from the email which I will give you. Extract data like origin, destionation, dates, timeframes, requested connection points (if specified explicitly) and ALL other flight information. Also, if there are any documents attached, read them too, they provide aditional information. You MUST read every single one of the attached documents, as they all include critical information.\n\nProvide an answer without asking me any further questions.\n\nEmail (in text format) to extract details from:\n\n" + emailText
        if len(imageInfo) > 0:
            content_text += "\n\nAlso take this important extra information about this email into consideration:\n" + imageInfo
    else:
        content_text = "Extract ALL flight details from the email which I will give you. Extract data like origin, destionation, dates, timeframes, requested connection points (if specified explicitly) and ALL other flight information.\n\nProvide an answer without asking me any further questions.\n\nEmail (in text format) to extract details from:\n\n" + emailText
    
    thread = client.beta.threads.create(
    messages=[
        {
        "role": "user",
        "content": content_text,
        "file_ids": files
        }
    ]
    )

    assistant_id=textFileAssistant.id

    run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant_id
    )

    while True:
        time.sleep(3)
        run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
        )
        print(run)
        print(run.status)

        if run.status == "failed":
            return "There was an error extracting data."
        if run.status == "completed":
            break
    
    print("Done")

    messages = client.beta.threads.messages.list(
    thread_id=thread.id
    )
    print("Answer:\n", messages.data[0].content[0].text.value)
    answer = messages.data[0].content[0].text.value
    apiDataHandler.delete_assistant(textFileAssistant.id, keys.openAI_APIKEY)

    for file_ in files:
        apiDataHandler.delete_file(file_, keys.openAI_APIKEY)

    return answer
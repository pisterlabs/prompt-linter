# import redis
# import os
# import platform
import openai 
import time
from pipedrive import PipedriveClient
from justcall import send_text
from datetime import datetime, timedelta

#improovy justcall info
improovy_api_key = "a475b0ecf1d1ba78ec7a9bc49d60f225531f3617"
improovy_api_secret = "bed00ba3e48de573ab7841e11971c32948edff6f"

# test = send_text('+16084205020', '+17736206534', 'testing1234', improovy_api_key, improovy_api_secret)
# if test == {'status': 'success'}:
#     print(True)

import requests

def query_calls(api_key, api_secret, per_page=1):
    headers = {
        "Accept": "application/json",
        "Authorization": f"{api_key}:{api_secret}"
    }

    data = {
        "type": '5',
        "per_page": per_page
    }

    url = "https://api.justcall.io/v1/calls/query"

    response = requests.post(url, headers=headers, json=data)
    return response.json()



def get_transcription(api_key, api_secret, id):
    headers = {
        "Accept": "application/json",
        "Authorization": f"{api_key}:{api_secret}"
    }

    params = {
        "id": id,
        "platform": 1
    }

    url = "https://api.justcall.io/v1/justcalliq/transcription"

    response = requests.get(url, headers=headers, json=params)
    return response.json()


def download_call_recording(api_key, api_secret, call_id, filename="recording.mp3"):
    headers = {
        "Accept": "application/json",
        "Authorization": f"{api_key}:{api_secret}"
    }

    data = {
        "id": call_id
    }

    url = "https://api.justcall.io/v1/calls/download"
    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Error {response.status_code}: {response.text}")

def add_voicemails():
    voicemail_body = query_calls(improovy_api_key, improovy_api_secret, 10)

    voicemails = voicemail_body['data']
    print('number of voicemails retrieved: ' + str(len(voicemails)))
    current_time = datetime.utcnow()
    time_limit = current_time - timedelta(minutes = 10)

    filtered_voicemails = []

    for voicemail in voicemails:
        voicemail_time_str = voicemail['time_utc']
        voicemail_time = datetime.strptime(voicemail_time_str, '%Y-%m-%d %H:%M:%S')

        if voicemail_time > time_limit:
            filtered_voicemails.append(voicemail)

    # Now `filtered_voicemails` contains voicemails from the last 25 hours
    # You can iterate through it the same way as the original list


    count = 1
    print('number of voicemails in the last 10 minutes: ' + str(len(filtered_voicemails)))
    for voicemail in filtered_voicemails:
        print('pulling voicemail ' + str(count))
        print('voicemail ' + str(count) + ' pulled')
        url = voicemail['recording']
        response = requests.get(url)
        with open('downloaded_file' + str(count) + '.mp3', 'wb') as file:
            file.write(response.content)

        audio_file= open('downloaded_file' + str(count) + '.mp3', "rb")
        print('generating transcript for voicemail ' + str(count))
        try:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        except:
            transcript = {'text': 'text too short'}
        print(transcript['text'])
        print('#####')

        count += 1
        crm = PipedriveClient('0a267f5c893e31a9dca5ff9bb3d0397266f69443')
        contactids = crm.contactIDs_from_num(voicemail['contact_number'])
        dealids = []
        for contactid in contactids:
            print('personid')
            print(contactid)
            deals = crm.deals_from_personID(contactid)
            dealids = []
            for deal in deals['data']:
                dealids.append(deal['id'])
            add_note = crm.add_note(content = 'VOICEMAIL: ' + transcript['text'], personid = contactid)
            print('added note' + str(add_note))
        for dealid in dealids:
            print('dealid')
            print(dealid)
            add_note = crm.add_note(content = 'VOICEMAIL: ' + transcript['text'], dealid = dealid)
            print('added note ' + str(add_note))




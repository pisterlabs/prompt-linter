import json
import os
import time
from flask import stream_with_context

from flask import Blueprint, request, jsonify, Response, render_template, redirect, url_for
from twilio.rest import Client
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Stream, Record, Dial
import logging

import requests
from requests.auth import HTTPBasicAuth
from openai import OpenAI

load_dotenv()

account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
openai_api_key = os.environ["OPENAI_API_KEY"]
client = Client(account_sid, auth_token)

twilio_blueprint = Blueprint('twilio', __name__)
call_status_updates = {}


@twilio_blueprint.route("/incoming_call", methods=['GET', 'POST'])
def incoming_call():
    logging.info("Incoming call received")
    # Log the request data for debugging
    logging.info(f"Request values: {request.values}")

    response = VoiceResponse()
    stream = Record(url='wss://e7a1-73-92-205-118.ngrok-free.app')
    response.append(stream)
    response.say("This call is being recorded for quality and training purposes.", voice='alice')
    # Keep the call active
    response.pause(length=10)

    return str(response)


@twilio_blueprint.route("/make_call", methods=['GET', 'POST'])
def make_call():
    data = request.get_json()
    contactPhone = data.get('phone')
    accountId = data.get('account_id')
    contactId = data.get('contact_id')
    # remove any non-numeric characters from the phone number
    contactPhone = ''.join(i for i in contactPhone if i.isdigit())

    # call for external deployment
    # twiml_url = url_for('twilio.twiml_response', _external=True, contactPhone=contactPhone, accountId=accountId, contactId=contactId)

    # call for local deployment
    twiml_url = f"https://e7a1-73-92-205-118.ngrok-free.app/twiml?contactPhone={contactPhone}&accountId={accountId}&contactId={contactId}"
    print(twiml_url)
    call = client.calls.create(
        to='14087907053',
        from_='18449683560',
        url=twiml_url,
    )
    return jsonify({"call_sid": call.sid}), 200


@twilio_blueprint.route("/twiml", methods=['GET', 'POST'])
def twiml_response():
    # get contactPhone from request
    contactPhone = request.values.get('contactPhone', '')
    accountId = request.values.get('accountId', '')
    contactId = request.values.get('contactId', '')
    response = VoiceResponse()
    # external deployment
    # dial = Dial(record='record-from-answer',
    #             recordingStatusCallback=url_for('twilio.status_callback', _external=True, accountId=accountId, contactId=contactId))

    # local deployment
    dial = Dial(record='record-from-answer',
                recordingStatusCallback=f"https://e7a1-73-92-205-118.ngrok-free.app/status_callback?accountId={accountId}&contactId={contactId}")
    dial.number(str(contactPhone))
    response.append(dial)
    return Response(str(response), mimetype='text/xml')


@twilio_blueprint.route("/status_callback", methods=['POST'])
def status_callback():
    # from main import socketio
    recording_url = request.values.get('RecordingUrl', '')
    accountId = request.values.get('accountId', '')
    contactId = request.values.get('contactId', '')

    if recording_url:
        transcription = download_and_transcribe(recording_url)
        # Use the meeting_minutes function to get all analysis data
        meeting_analysis = meeting_minutes(transcription.text)
        # Save the analysis data to the database
        save_interaction_results(accountId, contactId, meeting_analysis)
        # Emit an event to the connected clients
        # socketio.emit('task_complete', {'message': 'Call Analysis Complete!', 'data': meeting_analysis})
        call_status_updates[accountId] = "Call Analysis Complete"

        return jsonify(meeting_analysis), 200

    return '', 400  # It's a good practice to return a client error status if conditions aren't met


# import time
# import logging

# @twilio_blueprint.route('/stream')
# def stream():
#     def generate():
#         yield "data: Test message\n\n"
#         time.sleep(5)
#
#     return Response(stream_with_context(generate()), mimetype='text/event-stream')


@twilio_blueprint.route('/stream')
def stream():
    def generate():
        last_data_time = time.time()

        while True:
            current_time = time.time()

            # Check for updates
            has_update = False
            for account_id, status in list(call_status_updates.items()):
                logging.info(f"Sending update for account {account_id}: {status}")
                yield f"data: {json.dumps({'message': status})}\n\n"
                del call_status_updates[account_id]
                last_data_time = current_time
                has_update = True

            # If there was no update, send a comment line if 55 seconds have passed
            if not has_update and (current_time - last_data_time) >= 55:
                logging.info("Sending keep-alive comment")
                yield ": keep-alive\n\n"
                last_data_time = current_time

            time.sleep(1)

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


def save_interaction_results(account_id, contact_id, analysis_results):
    """
    Save the results of the OpenAI analysis to the Interaction model.
    """
    from app.models import db, Interaction
    interaction = Interaction(
        interaction_type='call',
        description=analysis_results['abstract_summary'],
        key_points=analysis_results['key_points'],
        action_items=analysis_results['action_items'],
        sentiment=analysis_results['sentiment'],
        account_id=account_id,
        contactId=contact_id
    )
    db.session.add(interaction)
    db.session.commit()


def download_and_transcribe(recording_url):
    recording_url_with_extension = recording_url + ".mp3"
    response = requests.get(recording_url_with_extension,
                            auth=HTTPBasicAuth(account_sid, auth_token))
    if response.status_code == 200:
        recording_data = response.content
        print("Recording downloaded.")
        # Further processing of recording_data
        with open("audio.mp3", "wb") as audio_file:
            audio_file.write(recording_data)
        client = OpenAI()
        audio_file = open("audio.mp3", "rb")
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript

    else:
        print(f"Failed to download recording: {response.status_code}")


def openai_analysis(transcription, analysis_type):
    """
    Generic function to perform different analysis types using OpenAI.
    """
    system_message = {
        'abstract': "Summarize content you are provided with for a user to later review.",
        'key_points': "Identify and list the main points that were discussed or brought up.",
        'action_items': "Identify any tasks, assignments, or actions that were agreed upon.",
        'sentiment': "Analyze the sentiment of the following text."
    }

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_message[analysis_type]
            },
            {
                "role": "user",
                "content": transcription
            }
        ],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def meeting_minutes(transcription):
    """
    Run all analysis types and collect the results in a dictionary.
    """
    analysis_functions = {
        'abstract_summary': 'abstract',
        'key_points': 'key_points',
        'action_items': 'action_items',
        'sentiment': 'sentiment'
    }

    results = {}
    for key, analysis_type in analysis_functions.items():
        results[key] = openai_analysis(transcription, analysis_type)

    return results

import base64
import json
import nltk
import datetime

import pandas as pd
from typing import List, Dict

from flask import session, url_for, render_template, request, redirect
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from main_app import app
from processing.WhatsappChatFileDataConverter import WhatsAppChatFileDataConverter
from routes.register_email_taps import db
from routes.likely_historical_events import get_historical_events_by_date
from parsing.whatsapp import extract_text_spoken_by_author, extract_dates_in_chat

from parsing.whatsapp_driven_responses import mimic_style

import openai

openai.api_key = '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'
USER_ROOT_PATH = f'users/prototype_users/{session["user_email_key"]}'


def get_email_user_name():
    service = build('gmail', 'v1',
                    credentials=Credentials.from_authorized_user_info(json.loads(session['email_token'])))
    email_address = service.users().getProfile(userId='me').execute()['emailAddress']
    session['local_specific_email'] = email_address
    return email_address


def save_email_labels():
    # Get the user's email address
    service = build('gmail', 'v1',
                    credentials=Credentials.from_authorized_user_info(json.loads(session['email_token'])))
    # Get the list of labels for the user's account
    labels = service.users().labels().list(userId='me').execute()
    return labels


@app.route('/retrieve_emails_from_label/<label_id>')
def retrieve_emails_from_label(label_id):
    """
    Route gets email attachments from a label,
    and then saves them to a firestore database.
    :param label_id: This is the label id of the label.
    """
    # attempt to call credentials
    try:
        creds = Credentials.from_authorized_user_info(json.loads(session['email_token']),
                                                      scopes=['https://www.googleapis.com/auth/gmail.readonly'])
        service = build('gmail', 'v1', credentials=creds)
    except Exception as error:
        # Handle any errors that occur while getting the user's credentials
        print(f'Error getting user credentials: {error}')
        return

    # build the path where this is stored
    common_path = f"{USER_ROOT_PATH}/{get_email_user_name()}"

    # Get the list of messages with the specified label
    messages = service.users().messages().list(userId='me', labelIds=[label_id]).execute()

    if len(messages) == 0:
        raise Exception('There are no messages in this email label.')

    # Loop through the messages and get their attachments
    for message in messages['messages']:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()

        # If the message has any attachments, loop through them and store their
        # filename, size, and MIME type in the attachments list
        if 'parts' in msg['payload']:
            for part in msg['payload']['parts']:
                if 'filename' in part:
                    if 'attachmentId' in part['body']:
                        # get attachment metadata
                        attachment = {
                            'filename': part['filename'],
                            'mimeType': part['mimeType'],
                            'attachmentId': part['body']['attachmentId'],
                        }

                        # get attachment data
                        attachment_data = {
                            'attachmentId': part['body']['attachmentId'],
                            'data': base64.urlsafe_b64decode(
                                service.users().messages().attachments().get(userId='me',
                                                                             messageId=message['id'],
                                                                             id=part['body']['attachmentId'])
                                .execute()['data']
                            )
                        }

                        doc_ref = (
                            db.collection(
                                f"{common_path}/{label_id}")
                            .document(attachment['filename'])
                        )

                        attachment_ref = (
                            db.collection(
                                f"{common_path}/attachments")
                            .document(attachment['filename'])
                        )
                        doc_ref.set(attachment, merge=True)
                        attachment_ref.set(attachment_data, merge=True)
    return redirect(url_for('index'))


@app.route('/retrieve_harvested_attachments')
def retrieve_harvested_attachments():
    label_document_map = {}
    doc_collection = (db
                      .collection('users')
                      .document('prototype_users')
                      .collection('afiq@gmail.com')
                      .document('afiqhatta.ah@gmail.com')
                      .collections())

    for collection in doc_collection:
        label_document_map[collection.id] = [item.id for item in collection.get()]

    return render_template('harvest/harvested_attachments.html', doc_collection=label_document_map)


def retrieve_attachments_from_firestore() -> List[Dict]:
    """
    Retrieves all attachments from Firestore for the current user.
    The attachments are retrieved from a collection located at
        'users/prototype_users/{user_email_key}/{get_email_user_name()}/attachments'
    :return: A list of dictionaries containing the attachments data.
    """
    # get the path of where the attachments are
    common_path = f"{USER_ROOT_PATH}/{get_email_user_name()}/attachments"
    attachment_collection = db.collection(common_path).get()

    # throw if now attachments exist.
    if not attachment_collection:
        raise IndexError('There are no downloaded attachments from your email labels.')

    return attachment_collection


@app.route('/summarise_relationships_from_attachments')
def summarise_relationships_from_attachments():
    """
    Query text from attachments and then summarise
    the relationships agents have with one another.
    :return:
    """
    # relationship summarisation path
    relationship_summarization = db.collection(
        f"{USER_ROOT_PATH}/{get_email_user_name()}/relationships"
    )

    index = 0
    for item in retrieve_attachments_from_firestore():
        # we convert the text from the attachments into a string format.
        text = (WhatsAppChatFileDataConverter(item.to_dict()['data'], item.id)
                .get_chat_string_from_file())

        # limit the amount of tokens to within limits
        truncated_text = ' '.join(nltk.word_tokenize(text)[:int(2000 * 0.75)])

        # append the script into the prompt.
        total_prompt = f"""Describe the nature of the relationship between 
        the two characters in this whatsapp script below. Use their names in the description. 
        Highlight one memorable event that could have occurred. 
        {truncated_text}
         """

        # Create a Completion object using the GPT-3 model
        completion = openai.Completion.create(
            engine="text-davinci-002",
            prompt=total_prompt,
            max_tokens=500,
            n=1,
            stop=["."],
        )

        # Print the completed text
        completed_text = completion.choices[0].text
        completed_text_dict = {
            'text': completed_text,
            'source': item.id,  # record the source of this conversation
        }
        relationship_summarization.document(str(index)).set(completed_text_dict)
        index += 1
    return 0


@app.route('/retrieve_known_relationships')
def retrieve_known_relationships():
    relationship_summarization_strings = db.collection(
        f"{USER_ROOT_PATH}/{get_email_user_name()}/relationships"
    ).get()

    # check if relationships exist
    if len(relationship_summarization_strings) == 0:
        return 'Currently you do not have relationships analysed.'

    # iterate through each conclusion that we came to
    relationship_summary = {}
    for item in relationship_summarization_strings:
        relationship_summary[item.id] = item.to_dict()

    return render_template('conclusions/relationship_summaries.html', summary=relationship_summary)


@app.route('/register_speech_style_from_attachments/<author>')
def register_speech_style_from_attachments(author):
    """
    This function caches data required for speech style, and filters things out to
    we get the things that we need. It also stores a timestamp
    that contains when this data was recorded.
    :param author: The required author who's style we need to mimic.
    """
    batch = 0
    speech_style_data_path = db.collection(
        f"{USER_ROOT_PATH}/{get_email_user_name()}/speech_style"
    ).document(str(batch))
    total_text = []

    # extract text in attachments one by one
    for item in retrieve_attachments_from_firestore():
        text = (WhatsAppChatFileDataConverter(item.to_dict()['data'], item.id)
                .get_chat_string_from_file())
        total_text += extract_text_spoken_by_author(text, author)

    # upload the training data to firestore
    total_training_text = ', '.join(total_text)
    total_training_dictionary = {
        'batch_index': str(batch),
        'timestamp': datetime.datetime.now(),
        'total_training_text': total_training_text
    }
    speech_style_data_path.set(total_training_dictionary)
    return 1


@app.route('/register_important_dates_from_whatsapp')
def register_important_dates_from_whatsapp():
    batch = 0
    known_dates_path = db.collection(
        f"{USER_ROOT_PATH}/{get_email_user_name()}/known_dates"
    ).document(str(batch))
    dates = []

    # extract text in attachments one by one
    for item in retrieve_attachments_from_firestore():
        text = (WhatsAppChatFileDataConverter(item.to_dict()['data'], item.id)
                .get_chat_string_from_file())
        dates += extract_dates_in_chat(text)  # get dates from whatsapp files

    try:
        known_dates_path.set({
            'dates': list(set(dates))  # store unique dates
        })
    except:
        raise Exception('Unable to upload dates for some reason ')
    return redirect(url_for('get_possible_events_from_firestore'))


@app.route('/register_possible_events')
def register_possible_events():
    batch = 0
    known_dates = db.collection(
        f"{USER_ROOT_PATH}/{get_email_user_name()}/known_dates"
    ).document(str(batch)).get()

    likely_events = db.collection(
        f"{USER_ROOT_PATH}/{get_email_user_name()}/probable_events"
    )

    # redirect users to get dates
    if known_dates.to_dict() is None:
        return render_template('redirects/no_known_dates.html')

    date_strings = (known_dates.to_dict()['dates'])

    # Parse the date strings using the strptime function and extract the month and year
    date_tuples = [(datetime.datetime.strptime(date_string, '%d/%m/%Y').month,
                    datetime.datetime.strptime(date_string, '%d/%m/%Y').year)
                   for date_string in date_strings]

    # Use the set function to remove duplicates from the list of date tuples
    unique_date_tuples = list(set(date_tuples))

    unique_event_limit = 5
    for date_tuple in unique_date_tuples[:unique_event_limit]:
        events = get_historical_events_by_date(*date_tuple)  # add historical date to tuple
        if len(events) != 0:
            for event in events:
                doc = likely_events.document(str(event['year']) + str(event['month']) + str(event['day']))
                doc.set({'text': event['event']})

    return redirect(url_for('get_possible_events_from_firestore'))


@app.route('/get_possible_events_from_firestore')
def get_possible_events_from_firestore():

    likely_events = db.collection(
        f"{USER_ROOT_PATH}/{get_email_user_name()}/probable_events"
    )
    event_list = []
    for doc in likely_events.stream():
        event_list.append({
            'date': pd.to_datetime(doc.id, format='%y%m%d'),
            'event_description': doc.to_dict()['text']
        })

    return render_template('conclusions/likely_events.html', event_list=event_list)


@app.route('/communication_style', methods=['GET', 'POST'])
def communication_style():
    """
    This function is a route that
    reinterprets simple commands and spits them out in
    the style of the writer.
    :return:
    """

    path_for_script = relationship_summarization_strings = db.collection(
        f"{USER_ROOT_PATH}/{get_email_user_name()}/style_prompts"
    ).get()

    if request.method == 'POST':
        completed_text = mimic_style(request.form['sentence'])
        return render_template('conclusions/mimic_communication_style.html', mimicking_text=completed_text)
    else:
        return render_template('conclusions/mimic_communication_style.html', mimicking_text='')

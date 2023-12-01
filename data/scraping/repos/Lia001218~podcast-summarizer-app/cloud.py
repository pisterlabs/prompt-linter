# main.py (Cloud Function backend)

import os
import pickle
import sys
from typing import Optional
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import modal
import feedparser
import datetime
# from models import SubscriptionModel
import whisper
import pathlib
import spacy
import wikipedia
import json
import openai
import tiktoken

from dataclasses import dataclass

@dataclass
class Podcast:
    # name: str 
    rss: str



@dataclass
class SubscriptionModel:
    user_email: str 
    podcasts: list[Podcast]
    receive_suggestions: bool

volume = modal.NetworkFileSystem.persisted("job-storage-vol")

image = modal.Image.debian_slim().apt_install("git").pip_install("git+https://github.com/openai/whisper.git", "openai", "tiktoken", "sendgrid", "spacy", "wikipedia", "feedparser")

stub = modal.Stub("podcast-app", image=image)


MODEL_DIR = "/cache"

@stub.function()
def get_recent_podcast_links(rss_url, max_episodes=3):

    feed = feedparser.parse(rss_url)

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=7)
    sorted_episodes = sorted(feed.entries, key=lambda entry: entry.published_parsed, reverse=True)


    recent_episode_links = []

    for entry in sorted_episodes[:max_episodes]:
      pub_date = datetime.datetime(*entry.published_parsed[:6])

      if  start_date <= pub_date <= end_date:
        recent_episode_links.append(entry.link)

    return recent_episode_links

@stub.function()
def semantic_split(podcast_transcript: str ):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(podcast_transcript)
    sentences = [sent.text for sent in doc.sents]
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
    # Group sentences with overlap
    size = 5
    overlap = 2
    sent_idx = 0
    string_list = []
    while sent_idx + size < len(sentences):
        string_list.append(''.join(sentences[sent_idx:sent_idx + size]))
        sent_idx += (size - overlap)

    build_large_chuck_list = []
    chunck = ''
    for i in range(len(string_list)):

      if len(enc.encode(chunck)) < 15000:
        chunck += string_list[i]
      else :
        build_large_chuck_list.append(chunck)
        chunck = ''
      if not chunck == '':
        build_large_chuck_list.append(chunck)

    return build_large_chuck_list

@stub.function(secret=modal.Secret.from_name("secret-keys"))
def create_podcast_sumary(podcast_transcript):

  openai.api_key = os.environ["OPENAI_API_KEY"]
  instructPrompt = """
  summarizes in a few sentences the main idea of the following text it is important that when you refer to the text you refer to it as if it were a podcast.
  """

  chatOutput = ''
  if type(podcast_transcript) is str:
    print('no list')
    request = instructPrompt + podcast_transcript
    chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ]
                                            )
  else:
    print('list')
    instructPrompt = """
      You will receive two texts, the first one is a short summary of one paragraph and the second one is a longer paragraph,
      I need you to summarize both texts as if they were one and contain the essential of both. Make sure that
      whenever you are going to refer to the text you do it as if it were a podcast.
      """
    for i in podcast_transcript :
      request = instructPrompt + str(chatOutput) + '\n' + i
      chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ])
  return chatOutput.choices[0].message.content

@stub.function(secret=modal.Secret.from_name("secret-keys"))
def get_information_guest( podcast_transcript):
  
    if type(podcast_transcript) is str:
        request = podcast_transcript
    else :
        request = podcast_transcript[0]

    openai.api_key = os.environ["OPENAI_API_KEY"]
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "user", "content": request}],
        functions=[
        {
            "name": "get_podcast_guest_information",
            "description": "Get information on the podcast guest using their full name and the name of the organization they are part of to search for them on Wikipedia or Google",
            "parameters": {
                "type": "object",
                "properties": {
                    "guest_name": {
                        "type": "string",
                        "description": "The full name of the guest who is speaking in the podcast",
                    },
                    "guest_organization": {
                        "type": "string",
                        "description": "The full name of the organization that the podcast guest belongs to or runs",
                    },
                    "guest_title": {
                        "type": "string",
                        "description": "The title, designation or role of the podcast guest in their organization",
                    },
                },
                "required": ["guest_name"],
            },
        }
    ],
    function_call={"name": "get_podcast_guest_information"}
    )


    podcast_guest = ""
    podcast_guest_org = ""
    podcast_guest_title = ""
    response_message = completion["choices"][0]["message"]
    if response_message.get("function_call"):
        function_name = response_message["function_call"]["name"]
        function_args = json.loads(response_message["function_call"]["arguments"])
        podcast_guest=function_args.get("guest_name")
        podcast_guest_org=function_args.get("guest_organization")
        podcast_guest_title=function_args.get("guest_title")

    if podcast_guest_org is None:
        podcast_guest_org = ""
    if podcast_guest_title is None:
        podcast_guest_title = ""

    input = wikipedia.page(podcast_guest + " " + podcast_guest_org + " " + podcast_guest_title, auto_suggest=True)
    return input.summary

@stub.function(secret=modal.Secret.from_name("secret-keys"))
def get_highlights(podcast_transcript):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    instructPrompt = """
    List the five most important highlights of the text.
    """
    if type(podcast_transcript) is str:
        request = instructPrompt + podcast_transcript
    else:
        request = instructPrompt + podcast_transcript[0]

    chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                                messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                        {"role": "user", "content": request}
                                                        ]
                                                )

    highlights = chatOutput.choices[0].message.content
    return highlights

@stub.function()
def process_transcriptions(transcriptions: list[str]):
   enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
   for i in transcriptions:
    if len(enc.encode(i)) > 10000:
        podcast_transcript = semantic_split.remote(i)  
        podcast_summary = str(create_podcast_sumary.remote(podcast_transcript))
        podcast_guest = str(get_information_guest.remote(podcast_transcript))
        podcast_highlights = str(get_highlights.remote(podcast_transcript))
    else:
        podcast_summary = str(create_podcast_sumary.remote(i))
        podcast_guest = str(get_information_guest.remote(i))
        podcast_highlights = str(get_highlights.remote(i))
        
    return podcast_summary, podcast_guest, podcast_highlights

@stub.function(
    gpu='T4',
    network_file_systems={MODEL_DIR: volume},
)
def transcribe_with_whisper(podcast_names: list[str]):
    model_path = os.path.join(MODEL_DIR, "medium.pt")
    if os.path.exists(model_path):
        print ("Model has been downloaded, no re-download necessary")
    else:
        print ("Starting download of Whisper Model")
        whisper._download(whisper._MODELS["medium"], MODEL_DIR, False)
    model = whisper.load_model('medium', device='cuda', download_root=MODEL_DIR)

    result = []
    for filename in os.listdir('/content/podcast_file'):
        if filename in podcast_names:
            audio_file_path = os.path.join('/content/podcast_file', filename)
            result.append(model.transcribe(audio_file_path))

    return result

@stub.function(
    network_file_systems={MODEL_DIR: volume},
)
def load_subscription_from_directory(user_email: str):
    file_path = os.path.join(MODEL_DIR, f"{user_email}.pickle")
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as file:
                subscription = pickle.load(file)
                return subscription
        except Exception as e:
            print(e)

    return None


@stub.function(
    network_file_systems={MODEL_DIR: volume},
)
def generate_newsletter(subscription_model: Optional[SubscriptionModel]):
    if subscription_model is None:
        file_path = os.path.join(MODEL_DIR, "user_data.pickle")
        try:
            with open(file_path, "rb") as file:
                subscription_model = pickle.load(file)
        except Exception as e:
            print(e)
            return "Failed to load user data"

    # Generate newsletter content 
    podcasts = []
    for p in subscription_model.podcasts:
        podcasts.extend(get_recent_podcast_links.remote(p.rss))
    
    transcriptions = transcribe_with_whisper.remote(podcasts)
    podcast_summary, podcast_guest, podcast_highlights = process_transcriptions.remote(transcriptions)
    return subscription_model.user_email, podcast_summary, podcast_guest, podcast_highlights

@stub.function(
    secret=modal.Secret.from_name("secret-keys"),
    network_file_systems={MODEL_DIR: volume},
    schedule=modal.Period(days=7),
)
def send_newsletter():
    os.environ["MY_PASSWORD"]
    # sender_email = os.environ["EMAIL_USER"]  # Replace with your email
    # sendgrid_api_key = os.environ[
    #     "SENDGRID_API_KEY"
    # ]  # Replace with your SendGrid API key

    subject = "Personalized Podcast Newsletter"
    user_email, podcast_summary, podcast_guest, podcast_highlights = generate_newsletter.remote()

    newsletter_content = f"gests: {podcast_guest} \highlights: {podcast_highlights}\n summary {podcast_summary}"
    
    message = f"Hi,\n\nHere's your weekly podcast newsletter:\n\n{newsletter_content}\n\nEnjoy!"

    # email_message = Mail(
    #     from_email=sender_email,
    #     to_emails=user_email,
    #     subject=subject,
    #     plain_text_content=message,
    # )

    # try:
    #     sg = SendGridAPIClient(sendgrid_api_key)
    #     response = sg.send(email_message)
    #     if response.status_code == 202:
    #         return "Newsletter sent successfully"
    # except Exception as e:
    #     print("Email sending error:", str(e))
    return message


@stub.function(
    network_file_systems={MODEL_DIR: volume},
)
def subscribe(subscription_model: SubscriptionModel):
    file_path = os.path.join(MODEL_DIR, "user_data.pickle")
    try:
        with open(file_path, "wb") as file:
            pickle.dump(subscription_model, file)
        return True
    except Exception as e:
        print(e)
        return False

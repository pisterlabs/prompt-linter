import modal

def download_whisper():
  """
  Downloads the Whisper model and saves it to Container storage.
  """
  import whisper
  print ("Download the Whisper model")

  # Perform download only once and save to Container storage
  whisper._download(whisper._MODELS["medium"], '/content/podcast/', False)


stub = modal.Stub("corise-podcast-project")

image_dependencies = [
  "ffmpeg",
  #"git"
]

dependencies = [
  "feedparser",
  "https://github.com/openai/whisper/archive/9f70a352f9f8630ab3aa0d06af5cb9532bd8c21d.tar.gz",
  #"git+https://github.com/sanchit-gandhi/whisper-jax.git",
  #"jax[cuda]",
  "requests",
  "ffmpeg",
  "openai",
  "tiktoken",
  "google-api-python-client",
  "ffmpeg-python"
]

# Whisper Jax (faster than regular Whisper):
#  - https://github.com/sanchit-gandhi/whisper-jax.git
#  - Notebook: https://github.com/sanchit-gandhi/whisper-jax/blob/main/whisper-jax-tpu.ipynb
# 
# Dependencies:
#  - https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier
jax_releases_url = "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
whisper_url = "https://github.com/openai/whisper/archive/9f70a352f9f8630ab3aa0d06af5cb9532bd8c21d.tar.gz"

corise_image = modal.Image.debian_slim().apt_install(*image_dependencies).pip_install(*dependencies, extra_index_url=jax_releases_url).run_function(download_whisper)

@stub.function(image=corise_image, gpu="any", timeout=600)
def get_transcribe_podcast_with_jax(rss_url, local_path):
  """
  Downloads a podcast episode from the given RSS feed URL and transcribes it using the Whisper Jax model.

  Args:
  - rss_url (str): The URL of the RSS feed.
  - local_path (str): The local path to save the downloaded podcast episode.

  Returns:
  - dict: A dictionary containing the podcast title, episode title, episode image, and episode transcript.
  """
  print ("Starting Podcast Transcription Function")
  print ("Feed URL: ", rss_url)
  print ("Local Path:", local_path)

  # Read from the RSS Feed URL
  import feedparser
  intelligence_feed = feedparser.parse(rss_url)
  podcast_title = intelligence_feed['feed']['title']
  episode_title = intelligence_feed.entries[0]['title']
  episode_image = intelligence_feed['feed']['image'].href
  for item in intelligence_feed.entries[0].links:
    if (item['type'] == 'audio/mpeg'):
      episode_url = item.href
  episode_name = "podcast_episode.mp3"
  print ("RSS URL read and episode URL: ", episode_url)

  # Download the podcast episode by parsing the RSS feed
  from pathlib import Path
  p = Path(local_path)
  p.mkdir(exist_ok=True, parents=True)

  print ("Downloading the podcast episode")
  import requests
  with requests.get(episode_url, stream=True) as r:
    r.raise_for_status()
    episode_path = p.joinpath(episode_name)
    with open(episode_path, 'wb') as f:
      for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

  print ("Podcast Episode downloaded")

  from whisper_jax import FlaxWhisperPipline

  # Load model from saved location
  print ("Load the Whisper model")

  # instantiate pipeline
  pipeline = FlaxWhisperPipline("openai/whisper-large-v2")

  # Perform the transcription
  print ("Starting podcast transcription")

  # JIT compile the forward call - slow, but we only do once
  result = pipeline(local_path + episode_name)

  # Return the transcribed text
  print ("Podcast transcription completed, returning results...")
  output = {}
  output['podcast_title'] = podcast_title
  output['episode_title'] = episode_title
  output['episode_image'] = episode_image
  output['episode_transcript'] = result['text']
  return output

@stub.function(image=corise_image, gpu="any", timeout=600)
def get_transcribe_podcast(rss_url, local_path):
  """
  Downloads a podcast episode from the given RSS feed URL and transcribes it using the Whisper model.

  Args:
  - rss_url (str): The URL of the RSS feed.
  - local_path (str): The local path to save the downloaded podcast episode.

  Returns:
  - dict: A dictionary containing the podcast title, episode title, episode image, and episode transcript.
  """
  print ("Starting Podcast Transcription Function")
  print ("Feed URL: ", rss_url)
  print ("Local Path:", local_path)

  # Read from the RSS Feed URL
  import feedparser
  intelligence_feed = feedparser.parse(rss_url)
  podcast_title = intelligence_feed['feed']['title']
  episode_title = intelligence_feed.entries[0]['title']
  episode_image = intelligence_feed['feed']['image'].href
  for item in intelligence_feed.entries[0].links:
    if (item['type'] == 'audio/mpeg'):
      episode_url = item.href
  episode_name = "podcast_episode.mp3"
  print ("RSS URL read and episode URL: ", episode_url)

  # Download the podcast episode by parsing the RSS feed
  from pathlib import Path
  p = Path(local_path)
  p.mkdir(exist_ok=True, parents=True)

  print ("Downloading the podcast episode")
  import requests
  with requests.get(episode_url, stream=True) as r:
    r.raise_for_status()
    episode_path = p.joinpath(episode_name)
    with open(episode_path, 'wb') as f:
      for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

  print ("Podcast Episode downloaded")

  # Load the Whisper model
  import whisper

  # Load model from saved location
  print ("Load the Whisper model")
  model = whisper.load_model('medium', device='cuda', download_root='/content/podcast/')

  # Perform the transcription
  print ("Starting podcast transcription")
  result = model.transcribe(local_path + episode_name)

  # Return the transcribed text
  print ("Podcast transcription completed, returning results...")
  output = {}
  output['podcast_title'] = podcast_title
  output['episode_title'] = episode_title
  output['episode_image'] = episode_image
  output['episode_transcript'] = result['text']
  return output

@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_summary(podcast_transcript):
  """
  Summarizes the main points of a podcast using OpenAI's GPT-3 model.

  Args:
  - podcast_transcript (str): The transcript of the podcast episode.

  Returns:
  - str: A summary of the main points discussed in the podcast.
  """
  import openai

  instructPrompt = """
     Act as a podcast transcriber. Write a 500-word summary of the podcast, highlighting the main topics, speakers, and memorable insights.
     Include direct quotations where relevant.
  """

  request = instructPrompt + podcast_transcript

  chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ]
                                            )

  podcastSummary = chatOutput.choices[0].message.content

  return podcastSummary

@stub.function(image=corise_image, secrets=[modal.Secret.from_name("my-openai-secret"), modal.Secret.from_name("google-custom-search-api-key"), modal.Secret.from_name("google-custom-search-engine-id")])
def get_podcast_guest(podcast_transcript):
  """
  Retrieves information about the guest of a podcast episode using OpenAI's GPT-3 model and Wikipedia.

  Args:
  - podcast_transcript (str): The transcript of the podcast episode.

  Returns:
  - dict: A dictionary containing information about the guest of the podcast episode. The dictionary has the following keys:
    - name (str): The name of the guest, along with their title and organization (if available).
    - summary (list): A list of dictionaries containing summary information about the guest, obtained from Google Custom Search API. Each dictionary in the list has the following keys:
      - title (str): The title of the search result.
      - link (str): The link to the search result.
      - snippet (str): A snippet of text from the search result.
  """
  import openai
  import json
  import os

  from googleapiclient.discovery import build

  request = podcast_transcript[:10000]

  completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
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

  print("Completion:", completion)

  podcast_guest = ""
  podcast_guest_org = ""
  podcast_guest_title = ""
  response_message = completion["choices"][0]["message"]
  
  if response_message.get("function_call"):
    function_args = json.loads(response_message["function_call"]["arguments"])
    podcast_guest=function_args.get("guest_name")
    podcast_guest_org=function_args.get("guest_organization")
    podcast_guest_title=function_args.get("guest_title")

  if podcast_guest_org is None:
    podcast_guest_org = ""
  if podcast_guest_title is None:
    podcast_guest_title = ""

  output = {
    "name": f"{podcast_guest} ({podcast_guest_title}, {podcast_guest_org})" if podcast_guest_title and podcast_guest_org else f"{podcast_guest} ({podcast_guest_title}{podcast_guest_org})" if podcast_guest_title or podcast_guest_org else podcast_guest,
    "summary": []
  }

  api_key = os.environ.get("GOOGLE_CUSTOM_SEARCH_API_KEY")
  cse_id = os.environ.get("GOOGLE_CSE_ID")
  query = f"{podcast_guest} {podcast_guest_org} {podcast_guest_title}"

  service = build("customsearch", "v1", developerKey=api_key)
  result = service.cse().list(q=query, cx=cse_id, num=10).execute()

  for item in result["items"]:
    title = item["title"]
    link = item["link"]
    snippet = item["snippet"]
    output["summary"].append({"title": title, "link": link, "snippet": snippet})

  return output

@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_highlights(podcast_transcript):
  """
  Given a podcast transcript, this function extracts highlights from the transcript. It captures unexpected or valuable takeaways for the reader and limits the results to 5 highlights.

  Args:
      podcast_transcript (str): The transcript of the podcast.

  Returns:
      str: The highlights extracted from the podcast transcript.
  """
  import openai
  instructPrompt = """
    Given a podcast transcript, I want you to extract highlights from the transcript. Capture unexpected or valuable takeaways for the reader. Limit the results to 5 highlights.
  """

  request = instructPrompt + podcast_transcript

  chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ]
                                            )
  chatOutput.choices[0].message.content

  podcastHighlights = chatOutput.choices[0].message.content
  return podcastHighlights

@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"), timeout=1200)
def process_podcast(url, path):
  """
  Processes a podcast by transcribing it, generating a summary, extracting guest information, and highlighting key moments.

  Args:
  - url (str): The URL of the podcast episode.
  - path (str): The path to the podcast episode file.

  Returns:
  - dict: A dictionary containing the processed podcast details, summary, guest information, and highlights.
  """
  output = {}
  podcast_details = get_transcribe_podcast.call(url, path)
  podcast_summary = get_podcast_summary.call(podcast_details['episode_transcript'])
  podcast_guest = get_podcast_guest.call(podcast_details['episode_transcript'])
  podcast_highlights = get_podcast_highlights.call(podcast_details['episode_transcript'])
  output['podcast_details'] = podcast_details
  output['podcast_summary'] = podcast_summary
  output['podcast_guest'] = podcast_guest
  output['podcast_highlights'] = podcast_highlights
  return output

@stub.local_entrypoint()
def test_method(url, path):
  """
  This function takes in a URL and path to a podcast episode and returns the podcast summary, guest information, and highlights.

  Args:
  - url (str): The URL of the podcast episode.
  - path (str): The path to the podcast episode.

  Returns:
  - None
  """
  podcast_details = get_transcribe_podcast.call(url, path)
  print ("Podcast Summary: ", get_podcast_summary.call(podcast_details['episode_transcript']))
  print ("Podcast Guest Information: ", get_podcast_guest.call(podcast_details['episode_transcript']))
  print ("Podcast Highlights: ", get_podcast_highlights.call(podcast_details['episode_transcript']))

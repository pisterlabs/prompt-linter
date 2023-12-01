import modal

def download_whisper():
  # Load the Whisper model
  import os
  import whisper
  print ("Download the Whisper model")

  # Perform download only once and save to Container storage
  whisper._download(whisper._MODELS["medium"], '/content/podcast/', False)


stub = modal.Stub("corise-podcast-project")
corise_image = modal.Image.debian_slim().pip_install("feedparser",
                                                     "https://github.com/openai/whisper/archive/9f70a352f9f8630ab3aa0d06af5cb9532bd8c21d.tar.gz",
                                                     "requests",
                                                     "ffmpeg",
                                                     "openai",
                                                     "tiktoken",
                                                     "wikipedia",
                                                     "ffmpeg-python").apt_install("ffmpeg").run_function(download_whisper)

@stub.function(image=corise_image, gpu="any", timeout=600)
def get_transcribe_podcast(rss_url, local_path):
  print ("Starting Podcast Transcription Function")
  print ("Feed URL: ", rss_url)
  print ("Local Path:", local_path)

  # Read from the RSS Feed URL
  import feedparser
  podcast_rss = feedparser.parse(rss_url)
  podcast_title = podcast_rss['feed']['title']
  episode_title = podcast_rss.entries[0]['title']
  episode_image = podcast_rss['feed']['image'].href
  for item in podcast_rss.entries[0].links:
    if (item['type'] == 'audio/mpeg'):
      episode_url = item.href
  episode_name = "podcast_episode.mp3"
  print ("RSS URL read and episode URL: ", episode_url)

  # Download the podcast episode by parsing the RSS feed
  from pathlib import Path
  p = Path(local_path)
  p.mkdir(exist_ok=True)

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
  import os
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
  import openai
  instructPrompt = """
  You will be provided with a podcast transcript about how to stay happy in life. You have to understand the content and context of the podcast
  and summarize it.
  Please provide a concise summary by using the provided podcast transcript.
  """

  request = instructPrompt + podcast_transcript
  chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ]
                                            )
  podcastSummary = chatOutput.choices[0].message.content

  return podcastSummary

@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_guest(podcast_transcript):
  import openai
  import wikipedia
  import json
  request = podcast_transcript[:10000]

  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": request}],
    functions=[
    {
        "name": "get_podcast_guest_information",
        "description": "Get information on the podcast guest using their full name",
        "parameters": {
            "type": "object",
            "properties": {
                "guest_name": {
                    "type": "string",
                    "description": "The full name of the guest speaker in the podcast",
                }
            },
            "required": ["guest_name"],
        },
    }
    ],
  function_call={"name": "get_podcast_guest_information"}
  )
  podcast_guest = ""
  response_message = completion["choices"][0]["message"]
  if response_message.get("function_call"):
    function_name = response_message["function_call"]["name"]
    function_args = json.loads(response_message["function_call"]["arguments"])
    podcast_guest=function_args.get("guest_name")
  
  print("Guest Name: ",podcast_guest)
  if not podcast_guest:
    return "Unknown"
  else:
    input = wikipedia.page(podcast_guest, auto_suggest=True)
    if input is not None:
      podcastGuest=input.summary
    else:
      print("Guest not found")
      podcastGuest = podcast_guest
      #podcastBlankGuestJson = '{"summary": "Summary unknown"}'
      #podcastGuest=json.loads(podcastBlankGuestJson,object_hook=lambda d: SimpleNamespace(**d))
  return podcastGuest

@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_highlights(podcast_transcript):
  import openai
  instructPrompt = """
  You will be provided with a podcast transcript. You have to understand the content and context of the podcast,
  summarize it and extract key moments or catchy insights from the guest or interesting conversation or questions asked during the podcast.

  Provide the result in the following format:

  Summary:

  Key Moments:
  """

  request = instructPrompt + podcast_transcript
  chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ]
                                            )
  podcastHighlights = chatOutput.choices[0].message.content
  return podcastHighlights

@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"), timeout=1200)
def process_podcast(url, path):
  output = {}
  podcast_details = get_transcribe_podcast.remote(url, path)
  podcast_summary = get_podcast_summary.remote(podcast_details['episode_transcript'])
  podcast_guest = get_podcast_guest.remote(podcast_details['episode_transcript'])
  podcast_highlights = get_podcast_highlights.remote(podcast_details['episode_transcript'])
  output['podcast_details'] = podcast_details
  output['podcast_summary'] = podcast_summary
  output['podcast_guest'] = podcast_guest
  output['podcast_highlights'] = podcast_highlights
  return output

@stub.local_entrypoint()
def test_method(url, path):
  output = {}
  podcast_details = get_transcribe_podcast.remote(url, path)
  print ("Podcast Summary: ", get_podcast_summary.remote(podcast_details['episode_transcript']))
  print ("Podcast Guest Information: ", get_podcast_guest.remote(podcast_details['episode_transcript']))
  print ("Podcast Highlights: ", get_podcast_highlights.remote(podcast_details['episode_transcript']))

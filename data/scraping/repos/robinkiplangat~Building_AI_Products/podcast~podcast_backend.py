import modal

def download_whisper():
  # Load the Whisper model
  import os
  import whisper
  print ("Download the Whisper model")

  # Perform download only once and save to Container storage
  whisper._download(whisper._MODELS["medium"], 'podcast/', False)


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
  model = whisper.load_model('medium', device='cuda', download_root='podcast/')

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
  ## ADD YOUR LOGIC HERE TO RETURN THE SUMMARY OF THE PODCAST USING OPENAI
  instructPrompt = """ You will be provided with a podcast transcript.
      Write a concise and effective summary that captures the key points, main arguments, and overall topics discussed in the episode.
      Please ensure You have understood the context and write:
      - Include a summary of the guest names and or speaker(s)
      - Highlight the key insights, takeaways, and conclusions from the podcast
      - Write in a clear, coherent, and engaging style that will capture a reader's interest
      - Keep the summary concise, aiming for around 200 to 250 words
      - Maintain neutrality and objectivity in your summary
      """

  ##
  import textwrap

  # Split the transcript into chunks that fit within the model's limits
  transcript_parts = textwrap.wrap(podcast_transcript, width=16000, break_long_words=False)

  # This list will store the summaries from each part of the transcript
  summaries = []

  for transcript_part in transcript_parts:
      request = instructPrompt + transcript_part
      chat_output = openai.ChatCompletion.create(
          model="gpt-3.5-turbo-16k",
          messages=[
              {"role": "system", "content": instructPrompt},
              {"role": "user", "content": request}
          ]
      )
      # Append each summary to the list
      summaries.append(chat_output.choices[0].message["content"])

  # Join summaries into one, forming the overall summary
  podcastSummary = ' '.join(summaries).strip()


  return podcastSummary

@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_guest(podcast_transcript):
  import openai
  import wikipedia
  import json
  ## ADD YOUR LOGIC HERE TO RETURN THE PODCAST GUEST INFORMATION
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

  podcast_guest = ""
  podcast_guest_org = ""
  response_message = completion["choices"][0]["message"]
  if response_message.get("function_call"):
    function_name = response_message["function_call"]["name"]
    function_args = json.loads(response_message["function_call"]["arguments"])
    podcastGuest=function_args.get("guest_name")
  return podcastGuest

@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_highlights(podcast_transcript):
  import openai
  ### ADD YOUR LOGIC HERE TO RETURN THE HIGHLIGHTS OF THE PODCAST
  instructPrompt = """
        please :summarize the key points and highlights from the podcast,
        Highlight the main discussion points, standout moments, and the overall messages conveyed in the podcast?
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
  output = {}
  podcast_details = get_transcribe_podcast.call(url, path)
  print ("Podcast Summary: ", get_podcast_summary.call(podcast_details['episode_transcript']))
  print ("Podcast Guest Information: ", get_podcast_guest.call(podcast_details['episode_transcript']))
  print ("Podcast Highlights: ", get_podcast_highlights.call(podcast_details['episode_transcript']))

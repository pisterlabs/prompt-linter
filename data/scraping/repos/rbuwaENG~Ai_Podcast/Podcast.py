from dotenv import load_dotenv
import os
import openai
import feedparser
import json
import requests

#configuration open ai and elevenlabs
load_dotenv()
openai.api_key = os.getenv("open_Apikey")
elevenlabs_api_key = os.getenv("elevenlab_apikey")

#rss feed setup
news_feed = "https://www.creepypasta.com/feed/"


print("## Processing RSS feed")

#feed parseer can send feeds
feed = feedparser.parse(news_feed)
stories = ""
stories_limit = 2

#adding title for each story and combile one long string
for item in feed.entries[:stories_limit]:
  stories = stories + " New Story: " + item.title + ". " + item.description

print("## Processing ChatGPT")
#assign chatgbt for summarized the content
#role is the user
#content is the command for chatgbt
chat_output = openai.chat.completions.create(
  model = "gpt-3.5-turbo",
  messages = [{
    "role": "user",
    "content": "Please rewrite the following horror movie titles and summaries as if they were being discussed on a one-off podcast.count characters and do not exceed 2500 character limit whe summarizig the story. The tone should be non-judgmental with no follow-up discussion, and end with a final closing greeting" + stories
  }]
)
chat_content = chat_output.choices[0].message.content

print(chat_content)

print("## Processing audio")
#converting to audio
voice_id = "21m00Tcm4TlvDq8ikWAM"
audio_output = requests.post(
  "https://api.elevenlabs.io/v1/text-to-speech/" + voice_id,
  data = json.dumps({
    "text": chat_content,
    "voice_settings": {
      "stability": 0.2,
      "similarity_boost": 0
    }
  }),
  headers = {
    "Content-Type": "application/json",
    "xi-api-key": elevenlabs_api_key,
    "accept": "audio/mpeg"
  }
)

if audio_output.status_code == 200:
  with open("test.mp3", "wb") as output_file:
    output_file.write(audio_output.content)
else:
  print(audio_output.text)

print("## Processing complete")



import genanki
import openai
import time
import random
from youtube_transcript_api import YouTubeTranscriptApi


# Replace 'YOUR_VIDEO_ID' with the ID of the YouTube video you want to download subtitles for
video_url = input("Enter the URL of the YouTube video you want to download subtitles for: ")
video_id = video_url.split("=")[1]
language_video = input("Enter the language of the subtitles you want to download: eg. en, fr, de, etc. ")
language_cards = input("Enter the language of the Anki cards you want to create")
size = input("Enter the number of cards you want to create")
openai.api_key = input("Enter your OpenAI API key: ")



try:
    # Fetch the transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_video])

except Exception as e:
    print(f"An error occurred: {e}")

#merge all the text in one string
text = ""
for i in transcript:
    text += i['text'] + " "
    

completion = openai.ChatCompletion.create(
      model="gpt-4", 
      messages = [{"role": "system", "content" : 
                  """
                  Condensate the following text into """ + size + """ questions and answers and build a python dictionary.
                  Return only the python dictionary. No variable name, just the dictionary. 
                  Like this: {"question1": "answer1", "question2": "answer2", ...}, since I will parse the dictionary from the string.
                  Do it in """ + language_cards + """.
                  """
                  
                  
                  },
                  {"role": "user", "content": text}
      ]
    )



# Create a Model for our Anki deck
my_model = genanki.Model(
  random.randint(0,99999999),  # Some unique number
  'Simple Model',
  fields=[
    {'name': 'Question'},
    {'name': 'Answer'},
  ],
  templates=[
    {
      'name': 'Card 1',
      'qfmt': '{{Question}}',
      'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
    },
  ])

# Italian political history facts
facts = eval(response)

# Generate Anki notes
notes = []
for q, a in facts.items():
    note = genanki.Note(
        model=my_model,
        fields=[q, a])
    notes.append(note)

# Create an Anki Deck and add the notes
my_deck = genanki.Deck(
  random.randint(0,99999999),  # Some unique number
  video_id+" Anki Deck")

for note in notes:
    my_deck.add_note(note)

# Generate the Anki deck file
genanki.Package(my_deck).write_to_file(f'{video_id}_deck.apkg')

print("Your Anki deck has been generated!")

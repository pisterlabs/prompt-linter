import openai
import json
import sys
import os
from dotenv import load_dotenv
load_dotenv()

# If OpenAI API key is stored in environment variable, uncomment below.
#openai.api_key = os.environ["OPENAI_KEY"]
# Otherwise, store it in the .env file
openai.api_key = os.getenv("OPENAI_KEY")

def create_flashcard(text_input):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=f'''
  Create an anki-style flashcard front and back from the following text.

  If it is not obvious, use your judgement to decide what it is that should be learnt from the flashcard, i.e. a fact, a definition, a concept, a quote. You can incorporate additional information about the topic that might be relevant.
  Your response should be in JSON format, comprising both a front and back. Here is the expected JSON format:
  {{"front": "front of flashcard", "back": "back of flashcard"}}.
  Here are some examples:
  Input: rank of a matrix
  Response: {{"front": "What is the rank of a matrix?", "back": "The rank of a matrix is the maximum number of linearly independent rows or columns in the matrix."}}

  Input: If the human brain were so simple that we could understand it, we would be so simple that we couldn’t.
  Response: {{"front": "Who said: 'If the human brain were so simple that we could understand it, we would be so simple that we couldn’t.'?", "back": "Emerson M. Pugh"}}

  Text to create flashcard from: {text_input}
''',
      temperature=0.8,
      max_tokens=500
    )

    return response.choices[0].text.strip()

input_text = sys.argv[1]
res = create_flashcard(input_text)

# Extract front and back from GPT-3 response, assuming it's in JSON format as requested
flashcard = json.loads(res)
front = flashcard['front']
back = flashcard['back']

# Escape special characters and add Remnote formatting
front = front.replace('"', '\\"')
back = back.replace('"', '\\"')
formatted_flashcard = f"{front} >> {back}"

# Call AppleScript command to add the flashcard to Remnote
os.system(f"""osascript -e 'tell application "Remnote" to activate
delay 1
tell application "System Events"
    tell process "Remnote"
        keystroke "/" using command down
        delay 0.5
        keystroke "ad"
        keystroke return
        delay 0.5
        keystroke "Flashcards"
        keystroke return
        delay 0.5
        keystroke "{formatted_flashcard}"
        delay 0.5
        key code 42 using command down -- navigate back to the title 
        delay 0.5
        keystroke "e" using {{option down, command down}}  -- merge note
    end tell
end tell' """)

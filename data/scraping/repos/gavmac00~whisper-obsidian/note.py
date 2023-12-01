import openai # imports whisper
import os # imports os
import re # imports re for sanitizing the title

from record import AudioRecorder # imports record.py

openai.api_key = os.getenv("OPENAI_API_KEY") # sets the API key

# take the title of the note
title = input("Note Title: ")

include_title = input("Include title in transcription? (y/n): ")
include_folder = input("Include folder to save transcription inside? (y/n): ")

if include_folder == "y":
    folder = input("Folder name: ")

# take in a stream of audio and save it as an audio file
if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.start()
    recorder.stop()

# define the audio file to be transcribed
audio_file= open("audio.wav", "rb")

# saves the response from the transcription (string)
response = openai.Audio.transcribe(
  "whisper-1", 
  audio_file,
  prompt=f"Transcribe the following audio recording titled:\n\n{title}\n\n into a well formatted note for the Obsidian software."
  )

transcript = response["text"]

obsidian_vault_path = "C:\\Users\\Gavin\\OneDrive\\Documents\\Obsidian Vault\\"

if include_title == "y":
  obsidian_note = f"# {title}\n\n{transcript}"
else:
  obsidian_note = f"{transcript}"

if include_folder == "y":
  obsidian_vault_path = f"{obsidian_vault_path}{folder}"
  if not os.path.exists(obsidian_vault_path):
    os.mkdir(obsidian_vault_path)

# Remove or replace invalid characters
sanitized_title = re.sub(r'[\\/*?:"<>|]', '_', title)

filepath = f"{obsidian_vault_path}/{sanitized_title}.md"

if os.path.exists(filepath):
  overwrite = input("File already exists. Append? (y/n): ")
  if overwrite == "y":
    existing_note = open(f"{obsidian_vault_path}/{sanitized_title}.md", "r")
    existing_note_text = existing_note.read()
    with open(f"{obsidian_vault_path}/{sanitized_title}.md", "w") as f:
      f.write(existing_note_text + "\n\n" + obsidian_note)
  else:
    print("File not saved.")
else:
  with open(f"{obsidian_vault_path}/{sanitized_title}.md", "w") as f:
      f.write(obsidian_note)

if include_folder == "y":
  print(f"Note saved as {sanitized_title}.md in Obsidian Vault\{folder}.")
else:
  print(f"Note saved as {sanitized_title}.md in Obsidian Vault.")
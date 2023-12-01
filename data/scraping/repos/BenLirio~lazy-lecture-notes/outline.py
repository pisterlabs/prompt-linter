import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


with open('transcript.txt') as f:
  transcript = f.read()

chunk_size = 8000
lecture_chunks = []
for i in range(0, len(transcript), chunk_size):
  lecture_chunks.append(transcript[i:i+chunk_size])

system = """
Your goal is to create an outline of a lecture given a lecture transcript.
I will provide you with a current outline (possibly empty) and a section of the lecture transcript.
Your task is to update the current outline with the new section of the lecture transcript.
This means that you may need to add or remove previous sections of the outline.
"""

def blockify(text):
  return f"\n\"\"\"\n{text}\n\"\"\"\n"

lecture_outline = ""
with open("outline.txt", "w") as lecture_notes_file:
  for i, lecture_chunk in enumerate(lecture_chunks):
    user = ""
    user += f"Outline: {blockify(lecture_outline)}"
    user += f"Transcript: {blockify(lecture_chunk)}"
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=[
        {"role": "system", "content": system },
        {"role": "user", "content": user }
      ]
    )
    lecture_outline = completion.choices[0].message.content
  lecture_notes_file.write(lecture_outline)
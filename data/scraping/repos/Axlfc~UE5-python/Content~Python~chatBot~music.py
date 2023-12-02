import openai
from dotenv import load_dotenv
import mido
import os
import json

def music():
    load_dotenv()
    openai.api_key = os.environ["OPENAI_API_KEY"]

    prompt_file = mido.MidiFile("test.mid")
    prompt_notes = []
    for track in prompt_file.tracks:
        for message in track:
            if message.type == "note_on" or message.type == "note_off":
                # Convert the message to a dictionary
                note = {
                    "type": message.type,
                    "note": message.note,
                    "velocity": message.velocity,
                    "time": message.time
                }
                prompt_notes.append(note)

    # Convert the list of dictionaries into a string
    prompt_string = json.dumps(prompt_notes)

    # Generate the musical composition
    completion = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt_string,
        temperature=0.5,
        max_tokens=1024,
        n=1,
        stop=None
    )

    # Retrieve and convert the generated composition
    response = completion.get("data")
    generated_notes = json.loads(response)
    # Save the generated MIDI file
    # Create a new MIDI file and add the generated notes
    generated_file = mido.MidiFile()
    generated_track = mido.MidiTrack()

    # Add the generated notes to the track
    for note in generated_notes:
        message = mido.Message.from_dict(note)
        generated_track.append(message)

    # Add the track to the MIDI file and save it
    generated_file.tracks.append(generated_track)
    generated_file.save("generated.mid")


def main():
    music()


if __name__ == '__main__':
    main()

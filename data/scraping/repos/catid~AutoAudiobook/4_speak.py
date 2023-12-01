import os, argparse, json, shutil

# OpenAI
import api_key
from openai import OpenAI
client = OpenAI(api_key=api_key.api_key)

def tts(text, output_filename, voice="nova"):
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice=voice,
        input=text,
    )

    response.stream_to_file(output_filename)


def generate_audiobook(args):
    if os.path.exists(args.speech_folder):
        shutil.rmtree(args.speech_folder)
    os.makedirs(args.speech_folder)

    actors = dict()

    with open(args.casting_file, 'r') as casting_file:
        content = casting_file.read()

        casting = json.loads(content)

        for role in casting:
            character = role["character"]
            voice = role["actor"]
            actors[character] = voice

    speech_index = 0

    for filename in os.listdir(args.assigned_folder):
        if filename.endswith('.json'):
            with open(os.path.join(args.assigned_folder, filename), 'r') as file:

                try:
                    content = file.read()

                    passage = json.loads(content)

                    for quote in passage:
                        character = quote["name"].lower()
                        text = quote["dialog"]
                        voice = actors[character]

                        speech_index += 1

                        output_filename = os.path.join(args.speech_folder, f'speech_{speech_index}.mp3')

                        print(f"{character}({voice}): {text}")

                        tts(text, output_filename, voice)

                except Exception as e:
                    print(f"Failed to process {filename}: {e}")


def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Speak all the parts.")
    parser.add_argument(
        '--assigned-folder', 
        type=str,
        nargs='?',  # Indicates the argument is optional
        default='assigned',  # Default folder name
        help="Folder of assigned JSON files."
    )
    parser.add_argument(
        '--casting_file', 
        type=str,
        nargs='?',  # Indicates the argument is optional
        default='casting.json',
        help="Casting file."
    )
    parser.add_argument(
        '--speech-folder', 
        type=str,
        nargs='?',  # Indicates the argument is optional
        default='speech',  # Default folder name
        help="Folder of output speech mp3."
    )

    # Parse the command line arguments
    args = parser.parse_args()

    generate_audiobook(args)

if __name__ == "__main__":
    main()

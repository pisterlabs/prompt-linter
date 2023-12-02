from dotenv import load_dotenv
from pathlib import Path
import openai
import json
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

tldrs_dir = Path("./tldrs")
tldrs_dir.mkdir(exist_ok=True)


def read_transcripts():
    transcripts_dir = Path("./transcripts")

    for file_path in transcripts_dir.glob("*.txt"):
        print(f"TLDR - {file_path}")

        tldr_path = Path(tldrs_dir, file_path.name)
        with open(file_path, "r") as file:
            text = file.read()

            # Edit prompt
            # response = openai.Edit.create(
            #     model="text-davinci-edit-001",
            #     input=text,
            #     instruction="tl;dr",
            # )

            # Completion prompt
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=text + "\n\nTl;dr",
                temperature=0.7,
                max_tokens=60,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=1,
            )
            print(response)

            with tldr_path.open("w") as file:
                try:
                    json.dump(response, file)
                except Exception as err:
                    print(err)

        return


if __name__ == "__main__":
    read_transcripts()
    print("Done")

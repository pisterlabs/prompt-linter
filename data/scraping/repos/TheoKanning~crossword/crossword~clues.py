import json
import os

import openai


def convert_raw_clues(raw_filename, output_filename):
    """
    Reads raw clue info from raw_filename, formats it to match GPT-3's fine-tune input, and writes it to output_filename

    Raw clues are formatted like "Up in the air : ALOFT"
    """

    with open(output_filename, "w+") as f_out:
        f_out.write("farts")

    with open(raw_filename, "r") as f_in:
        with open(output_filename, "w+") as f_out:
            for line in f_in.readlines():
                line = line.strip()
                if not line:
                    continue

                if line.isnumeric():
                    # This line is a clue number, ignore it
                    continue

                if line.lower() == "down" or line.lower() == "across":
                    continue

                components = line.rsplit(
                    ":", 1
                )  # split from end to handle colons inside clues
                if len(components) != 2:
                    print(line)
                    continue

                clue = components[0].strip()
                answer = components[1].strip()

                f_out.write(
                    json.dumps(
                        {
                            "prompt": f"Answer: {answer.lower()}\nClue:",
                            "completion": f" {clue}\n",
                        }
                    )
                )
                f_out.write("\n")


def get_clue(answer):
    prompt = f"Answer: {answer.lower()}\nClue:"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    result = openai.Completion.create(
        model="curie:ft-personal-2022-04-30-18-38-57", prompt=prompt, stop="\n", n=5
    )
    print(f"Answer: {answer}\nClues:")
    for choice in result["choices"]:
        print(choice["text"])


if __name__ == "__main__":
    get_clue("")
    # convert_raw_clues("../clues/raw_clues.txt", "../clues/formatted.jsonl")

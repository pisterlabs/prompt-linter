#!/usr/bin/env python

__doc__ = """
Usage: run_tower_gpt.py [options]

Options:
    -s, --sketch <sketch>  Sketch to use
"""

from docopt import docopt
import openai
import os
import subprocess
import json


cur_dir = os.path.dirname(os.path.realpath(__file__))
prompt_file = cur_dir + "/../" + ("tower_prompt.txt")


def main(args):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ["OPENAI_ORGANIZATION"]

    with open(prompt_file, "r") as f:
        prompt = f.read()

    prompt += "\nOutput:\n"
    cmd = [cur_dir + "/../_build/default/bin/print_tower.exe"]
    prompt += subprocess.check_output(
        cmd,
        encoding="utf-8",
    )
    prompt += "\nProgram:\n"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        n=10,
    )
    progs = [r["message"]["content"] for r in response.choices]
    with open("full_prompt", "w") as f:
        f.write(prompt)
    print(json.dumps(progs))
    # print(
    #     subprocess.check_output(
    #         ["_build/default/bin/print_bench.exe", "-scaling", "2"],
    #         input=prog,
    #     ).decode("utf-8")
    # )


if __name__ == "__main__":
    main(docopt(__doc__))

import os
import openai
import dotenv
from argparse import ArgumentParser
import pickle
import json
import os
import time


PROMPT_HELLASWAG = """Please generate 5 different sentences that could have started the following documents:
{docs}
Format the results as a JSON in the following format:
    "prompts": [
        "example beginning sentence1",
        "example beginning sentence2",
        "example beginning sentence3",
        "example beginning sentence4",
        "example beginning sentence5",
    ]
Make sure to only include 5 different sentences in the response.
"""

PROMPT_INSTRUCT = """Please generate 5 different prompts that could have created the following documents, and please make sure to generate the responses as JSON only and keep the prompts brief:
{docs}

Here is an example for a set of documents about cooking steak:

{{
    "prompts": [
        "What is a good recipe for steak?",
        "Give me a steak dinner recipe.",
        "Tell me how to cook steak",
        "What's a good way to make a steak?",
        "What is the best recipe for fast steak?",
    ]
}}

Simply provide JSON in the following above format. Do not provide any additional text that deviates from the format specified in the example.
"""

# Make sure to format the result ONLY as a JSON object only so that it can be easily parsed in Python. Keep the prompts brief and not too long.


if __name__ == "__main__":
    dotenv.load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.Model.list()

    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_path", type=str)
    # parser.add_argument("--dataset_type", type=str)

    args = parser.parse_args()

    dataset = pickle.load(open(args.dataset_path, "rb"))
    results = []

    for cur in dataset:
        cur_docs = ""
        for d in cur["train_docs_str"]:
            cur_docs += d + "\n\n"

        response = None
        json_error = False
        while response == None:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "user",
                            "content": PROMPT_INSTRUCT.format(docs=cur_docs),
                        }
                    ],
                )

                response = json.loads(response.choices[0].message.content)
                if "prompts" in response:
                    response = response["prompts"]
                elif "responses" in response:
                    response = response["responses"]

                results.append(
                    {
                        "id": cur["id"],
                        "prompt": cur["prompt"],
                        "responses": response,
                    }
                )
            except Exception as e:
                print(e)
                if isinstance(e, json.decoder.JSONDecodeError):
                    response = None
                else:
                    time.sleep(10)

        json.dump(
            results,
            open(os.path.join(args.output_path, "gpt_suggested.json"), "w"),
            indent=4,
            ensure_ascii=False,
        )

import json
from time import sleep
from tqdm import tqdm
import openai
import os

from templates import sentiment_likert_prompt, helpfulness_likert_prompt

MAX_API_RETRY = 5
REQ_TIME_GAP = 5

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_eval(user_prompt: str, function_name: str):
    sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer."
    functions = [
        {
            "name": "sentiment_likert",
            "description": "Extracts the sentiment score of the response, plus a reasoning for that score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "integer",
                        "description": "The score of the sentiment, from 1 to 5.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "The reasoning for the score.",
                    },
                },
                "required": ["sentiment", "reasoning"],
            },
        },
        {
            "name": "helpfulness_likert",
            "description": "Extracts the helpfulness score of the response, plus a reasoning for that score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "integer",
                        "description": "The score of the helpfulness, from 1 to 6.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "The reasoning for the score.",
                    },
                },
                "required": ["score", "reasoning"],
            },
        },
    ]
    for _ in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
                function_call={"name": function_name},
                functions=functions,
            )

            return json.loads(
                response["choices"][0]["message"]["function_call"]["arguments"]
            )

        except Exception as e:
            print(e)
            sleep(REQ_TIME_GAP)

    raise Exception(f"Failed after {MAX_API_RETRY} retries.")


def likert_score(prompts, outputs, prompt_template, function_name):
    if isinstance(prompts, str) and isinstance(outputs, str):
        prompts = [prompts]
        outputs = [outputs]

    assert len(prompts) == len(outputs), "Number of prompts and outputs must be equal"

    data = []
    for prompt, output in tqdm(zip(prompts, outputs)):
        critique_dict = get_eval(
            prompt_template.format(task=prompt, submission=output),
            function_name=function_name,
        )
        data.append(critique_dict)

    return data


if __name__ == "__main__":
    print("Sentiment Likert")
    scores = likert_score(
        prompts=["Evaluate the sentiment of this movie review"] * 3,
        outputs=[
            "The film was simply amazing. The acting was great, the plot was interesting, and the cinematography was beautiful. I would recommend this movie to anyone who enjoys a good drama.",
            "I'm not sure if I liked this movie. It was a bit too long and the plot was confusing. The acting was good, but the cinematography was a bit too dark. I would recommend this movie to anyone who enjoys a good drama.",
            "What a waste of time. The acting was terrible, the plot was boring, and the cinematography was awful. I would not recommend this movie to anyone.",
        ],
        prompt_template=sentiment_likert_prompt,
        function_name="sentiment_likert",
    )
    print(scores)
    print("Helpfulness Likert")
    scores = likert_score(
        prompts=["What is the capital of France?"] * 3,
        outputs=[
            "Paris is the capital of France",
            "Paris, but I'm not sure",
            "Madrid",
        ],
        prompt_template=helpfulness_likert_prompt,
        function_name="helpfulness_likert",
    )
    print(scores)

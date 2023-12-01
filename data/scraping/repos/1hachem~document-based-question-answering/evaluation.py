import asyncio
import os

import dotenv
import lmql
import openai

from src.lm import GPT2

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@lmql.query
async def evaluate_gpt3(prediction: str, ground_truth: str):
    '''argmax
        """output 1 if the student answer is similar to true answer, 0 otherwise
        ignore small differences
        student answer: {prediction}
        true answer: {ground_truth}
        evaluation : [EVALUATION]"""
    from
        "openai/text-davinci-003"
    distribution
        EVALUATION in ["1", "0"]
    '''


@lmql.query
async def evaluate_gpt2(prediction: str, ground_truth: str):
    '''argmax
        """you are a correction model you output 1
        if a given student answer the same as the true answer,
        you output 0 otherwise
        example :
        student answer: a humbugger
        true answer: a toast
        evaluation : 0
        student answer: the ratio is 20:1
        true answer: 20 to 1
        evaluation : 1
        student answer: {prediction}
        true answer: {ground_truth}
        evaluation : [EVALUATION]"""
    from
        "gpt2-large"
    distribution
        EVALUATION in ["1", "0"]
    '''


async def evaluate_chatgpt(predictions: list[str], gts: list[str]):
    conversation = [
        {
            "role": "system",
            "content": """output 1 if the student answer is the same as true answer, 
            0 otherwise ignore case and small differences""",
        }
    ]
    evaluations = []
    for prediction, ground_truth in zip(predictions, gts):
        conversation.append(
            {
                "role": "user",
                "content": f"""student answer: {prediction} 
                true answer: {ground_truth}""",
            }
        )
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=conversation,
            max_tokens=1,
            logit_bias={
                "16": 100,
                "15": 100,
            },  # 16 and 15 are the ids of 1 and 0 respectively
        )
        evaluations.append(completion.choices[0].message.content)

    return evaluations


async def is_correct(prediction: str, ground_truth: str):
    evaluation = await evaluate_gpt3(prediction, ground_truth)
    evaluation = evaluation.variables["EVALUATION"]
    if evaluation == "1":
        return 1
    else:
        return 0


if __name__ == "__main__":
    # evaluation = asyncio.run(evaluate_gpt2("experiment", "test"))
    evaluation = asyncio.run(is_correct("yup", "yes"))
    print(evaluation)

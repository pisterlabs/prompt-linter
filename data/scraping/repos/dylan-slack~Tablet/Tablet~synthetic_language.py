"""Compute synthetic language revisions."""
from datetime import datetime, timezone, time

import numpy as np
from time import sleep

import openai


def load_openai_key(path: str):
    with open(path, "r") as k:
        op_k = k.read().replace("\n", "")
    openai.api_key = op_k


def create_revision_instructions(description: str, instruction_type: str, context: str):
    if instruction_type == "prototypes":
        instruction = """You are provided with examples of prototypical datapoints for each class in a dataset. For each prototypical data point, you are provided with the features for that data point, feature values, and corresponding class. You must revise these prototypical data points and corresponding class values into a human readable paragraph. This paragraph should describe, in high level terms, the patterns in the prototypical data points associate with different class values. This paragraph will be used for humans to make decisions surrounding new data points, so your summary should be clearly readable and present actionable information a person can use to classify new data points in the future."""
        instruction += f"\n#####\nPrototypical Datapoints & Context: {context}\n{description}\n#####\nSummary:"
    elif instruction_type == "ruleset":
        instruction = """You are provided with examples of rules for predicting each class in a dataset. Each rule applies to a single class and gives you a rule for predicting that class. Each rule also has a confidence level. You must revise these rules into a human readable paragraph. This paragraph should describe, in high level terms, the patterns in the rules associated with different class values. This paragraph will be used for humans to make decisions on new data points, so your summary should be clearly readable and present actionable information a person can easily use to classify new data points."""
        instruction += f"\n#####\nRules & Context: {context}\n{description}\n#####\nSummary:"
    else:
        raise ValueError(f"Don't have instruction type {instruction_type}")
    return instruction


def strip_text(gpt3_response):
    responses = [gpt3_response["choices"][i]["text"] for i in range(len(gpt3_response["choices"]))]
    return responses


def get_gpt3_revisions(text: str, n: int, model: str = "text-davinci-003", key_path: str = "oai-key.txt"):
    print("Launching openai completions...")
    load_openai_key(key_path)
    result = []
    for _ in range(n):
        temp = np.random.choice([0.2, 0.95])
        pp = np.random.choice([0.2, 2.0])
        did_it = False
        while not did_it:
            try:
                response = openai.Completion.create(
                    model=model,
                    prompt=text,
                    temperature=temp,
                    max_tokens=512,
                    top_p=1,
                    frequency_penalty=0.0,
                    n=1,
                    presence_penalty=pp
                )
                did_it = True
            except Exception as e:
                print(f"Failed with {e}...")
                sleep(10)
        rtext = strip_text(response)
        result.extend(rtext)
    print("Got completions...")
    return result


message = """You are a helpful assistant."""


def get_trad_prompt(data):
    new_end = ("Please answer with yes or "
               "no. For example, if you "
               "believe the answer is yes, answer 'The answer "
               "is yes'. If you think the answer is no, respond with "
               "'The answer is no'.")

    prompt = data["text"].replace("The answer is yes",
                                  new_end + "\nThe answer is yes")
    prompt = prompt.replace("\nThe answer is no",
                                  "\n" + new_end + "\nThe answer is no")
    prompt = prompt.replace("Now complete the following example - Input: ", "\n")
    prompt = prompt[:-len("The answer is")] + new_end
    messages = [
        {"role": "system", "content": message},
        {"role": "user", "content": prompt},
    ]
    done = False
    while not done:
        try:
            results = openai.ChatCompletion.create(
                top_p=0.1,
                max_tokens=16,
                model="gpt-3.5-turbo",
                messages=messages
            )
            done = True
        except Exception as mye:
            print(f"Failed with {mye}")
            np.random.seed(int(datetime.now(timezone.utc).timestamp() * 100) % (2 ** 32 - 1))
            slp_time = np.random.choice(list(range(30)))
            print(f"Sleeping for {slp_time}")
            sleep(slp_time)

    pred = results["choices"][0]["message"]["content"]
    return pred, results["usage"]["total_tokens"]


def chatgpt_predict(data: dict, key_path: str = "oai-key.txt",
                    max_tokens: int = 1, lift: bool = False,
                    trad_prompt: bool = False):
    """Chat-gpt prediction.

    trad_prompt determines whether to use a traditional LLM prompt if true. Otherwise, we experiment with trying to
    format the prompts into more of a conversation. This ended up not working that well though.
    """
    load_openai_key(key_path)
    if trad_prompt:
        return get_trad_prompt(data)

    data["serialization"] = data["serialization"].replace("::", ":")
    if len(data["k_shot_samples"]) > 0:
        k_shot = data["k_shot_samples"].split("\n\n\n")
        k_shot_chatgpt = [
            {"role": "user", "content": "I am going to give you examples of patients and whether the disease "
                                        "should be included in the "
                                        "differential diagnosis. Pay close attention to the language the "
                                        "patients use and remember it when you go to classify new patients! If any of the "
                                        "diagnoses are surprising to you, make sure to remember the symptoms the patient has "
                                        "and look out for them in the future."},
            {"role": "system", "content": "Okay, thank you for showing me these, and I understand using these "
                                          "examples is extremely important. Show me the examples."}
        ]

        ap = (f"\nAs a reminder, here is your task: {data['lift_header']} "
              f"Should this disease be included the "
              "differential diagnosis? "
              "Please answer with your reasoning followed by yes or "
              "no. For example, if you "
              "believe the answer is yes, answer with your "
              "rationale followed by 'The answer "
              "is yes'. If you think the answer is no, respond with "
              "your rationale followed by "
              "'The answer is no'.")
        for k in k_shot:
            ll = k.split("\n")
            prompt = "\n".join(ll[:-1])
            # answer = "Based on the patient's symptoms, " + ll[-1].lower() + "."
            answer = ll[-1]
            nmessageu = {"role": "user", "content": k}
            nmessagea = {"role": "system", "content": f"I understand the answer is: `{answer}`"}
            k_shot_chatgpt += [nmessageu] + [nmessagea]
        k_shot_chatgpt += [{"role": "user", "content": "Now, using these examples, let's look at a new patient."},
                           {"role": "system", "content": "Okay, let's do it."}]
    else:
        print("Not building kshot...")
        k_shot_chatgpt = []

    if lift:
        print("Using lift...")
        messages = [
                       {"role": "system", "content": message},
                       {"role": "user", "content": data["lift_header"]},
                       {"role": "assistant", "content": "I understand."}] \
                   + k_shot_chatgpt + \
                   [{"role": "user",
                     "content": data["serialization"] + f"\nAs a reminder, here is your task: {data['lift_header']} "
                                                        f"Should this disease be included the "
                                                        "differential diagnosis? "
                                                        "Please answer with your reasoning followed by yes or "
                                                        "no. For example, if you "
                                                        "believe the answer is yes, answer with your "
                                                        "rationale followed by 'The answer "
                                                        "is yes'. If you think the answer is no, respond with "
                                                        "your rationale followed by "
                                                        "'The answer is no'."},
                    ]
    else:
        print("Using instructions...")
        messages = [
                       {"role": "system", "content": message},
                       {"role": "user", "content": data["header"]},
                       {"role": "assistant", "content": "I understand. Please tell "
                                                        "me the instructions."},
                       {"role": "user", "content": data["instructions"]},
                       {"role": "assistant", "content": "Got it, that makes sense. I will follow these instructions."}] \
                   + k_shot_chatgpt + \
                   [{"role": "user",
                     "content": data["serialization"] + f"\nAs a reminder, here is your task: {data['header']} "
                                                        f"Should this disease be included the "
                                                        "differential diagnosis? "
                                                        "Please answer with your reasoning followed by yes or "
                                                        "no. For example, if you "
                                                        "believe the answer is yes, answer with your "
                                                        "rationale followed by 'The answer "
                                                        "is yes'. If you think the answer is no, respond with "
                                                        "your rationale followed by "
                                                        "'The answer is no'."},
                    ]
    done = False
    while not done:
        try:
            results = openai.ChatCompletion.create(
                top_p=0.1,
                max_tokens=128,
                model="gpt-3.5-turbo",
                messages=messages
            )
            done = True
        except Exception as mye:
            print(f"Failed with {mye}...")
            np.random.seed(int(datetime.now(timezone.utc).timestamp() * 100) % (2 ** 32 - 1))
            slp_time = np.random.choice(list(range(10)))
            print(f"Sleeping for {slp_time}...", flush=True)
            sleep(slp_time)
    pred = results["choices"][0]["message"]["content"]
    return pred, results["usage"]["total_tokens"]

import openai
import json
from imagenet_prompts import fgvc_aircraft
from tqdm import tqdm
from pathlib import Path

openai.api_key = "PLEASE ENTER YOUR API KEY HERE"

all_json_dict = {}
all_responses = {}

category_list_all = {
    'FGVCAircraft': fgvc_aircraft
}


vowel_list = ['A', 'E', 'I', 'O', 'U']



for k, v in category_list_all.items():
    print('Generating descriptions for ' + k + ' dataset.')

    json_name_all = f"tap/{k}.json"

    if Path(json_name_all).is_file():
        raise ValueError("File already exists")

    for i, category in enumerate(tqdm(v)):
        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"

        if '_' in category:
            category = category.replace('_', ' ')

        prompts = []
        prompts.append("Describe how does the " + category + " aircraft looks like.")
        prompts.append("How can you recognize the aircraft " + category + "?")
        prompts.append("What does the aircraft " + category + " look like?")
        prompts.append("Describe an image of the aircraft " + category + ".")
        prompts.append("How can you identify the " + category + " aircraft:")
        all_result = []
        for curr_prompt in prompts:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=curr_prompt,
                temperature=.99,
                max_tokens=50,
                n=5,
                stop="."
            )

            for r in range(len(response["choices"])):
                result = response["choices"][r]["text"]
                all_result.append(result.replace("\n\n", "") + ".")

        all_responses[category] = all_result

        # if i % 10 == 0:
    with open(json_name_all, 'w') as f:
        json.dump(all_responses, f, indent=4)

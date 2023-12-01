from argparse import ArgumentParser, Namespace
import json
import os
from tqdm import tqdm
import time
import openai


''' Read the prompt template. '''
def get_prompt_template(path: str):

    with open(path, "r") as fp:
        template = fp.read()
    
    return template


''' Add attributes of the target speech to prompt template. '''
def add_attributes(attributes: list):

    ''' 
    Input: List

        attributes: ["High pitch", "Fast speed", "Low volume", "Sad emotion"]
    
    Return: String

        input_attributes: "High pitch, Fast speed, Low volume, Sad emotion."    
    '''

    attributes_num = len(attributes)

    input_attributes = ""
    for index, attribute in enumerate(attributes):

        if index < (attributes_num - 1):
            input_attributes += f"{attribute}, "
        else:
            input_attributes += f"{attribute}."
    
    return input_attributes


''' Combine prompt template and current input attributes of target speech. '''
def format_prompt(prompt_template, input_attributes, num_style_prompt):

    prompt_template = prompt_template.replace("[Provide the attributes you want transformed here]", input_attributes) 
    prompt_template = prompt_template.replace( "[The number of style prompt]", str(num_style_prompt)) 

    return prompt_template


''' Handle the query of gpt-4. '''
def query(messages, model="gpt-4", n=1, temp=0.8):

    while True:
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=n,
                temperature=temp,
            )
            break

        except:
            print("\nChatGPT API failed. Try again.\n")
            time.sleep(1)
            continue

    return completion


''' Handle the query message and response of gpt-4. '''
def handle_query(messages, model, n, temp, result_path=None):

    message = [{"role": "user", "content": messages}]

    response = query(message, model, n, temp)

    parameters = {"model": model, "n": n, "temp": temp}

    result = {
        "request": message,
        "response": response,
        "params": parameters,
    }

    return result


''' Return style prompt. '''
def get_style_prompt(prompt_template_path, attributes, num_style_prompt, output_response_path, model="gpt-4"):

    prompt_template = get_prompt_template(path=prompt_template_path)

    format_attributes = add_attributes(attributes=attributes)

    prompt = format_prompt(prompt_template=prompt_template, input_attributes=format_attributes, num_style_prompt=num_style_prompt)

    response = handle_query(messages=prompt, model=model, n=1, temp=1)

    with open(output_response_path, "w") as fp:
        json.dump(response, fp, indent=4)

    print(response)

    return response


''' Parse the response from gpt-4 to a list of style prompt. '''
def parse_response(response):

    # with open("/home/u2619111/frank/data_augmentation/response.json", "r") as fp:
    #     response = json.load(fp)

    contents = response["response"]["choices"][0]["message"]["content"]
    contents = contents.split("\n")

    style_prompt_list = []
    
    for content in contents:

        # remove index.
        style_prompt_list.append(content[3:])

    return style_prompt_list

    
def parse_args() -> Namespace:

    parser = ArgumentParser()
    parser.add_argument("-k", "--key", type=str, default="your-openai-key")
    parser.add_argument("-p", "--prompt_template", type=str, default="./prompt_template.txt")
    parser.add_argument("-n", "--num_style_prompt", type=str, default=5)
    parser.add_argument("-o", "--output_resonse_path", type=str, default="./response.json")

    args = parser.parse_args()

    return args


def main(args):

    start = time.time()

    openai.api_key = args.key
    prompt_template_path = args.prompt_template
    num_style_prompt = args.num_style_prompt
    output_response_path = args.output_resonse_path

    attributes = ["High pitch", "Low volume", "Fast speed", "Sad emotion"]

    response = get_style_prompt(prompt_template_path, attributes, num_style_prompt=num_style_prompt, output_response_path=output_response_path, model="gpt-4")
    style_prompt_list = parse_response(response=response)

    print(style_prompt_list)

    print(time.time() - start)


if __name__ == "__main__":
    args = parse_args()
    main(args)
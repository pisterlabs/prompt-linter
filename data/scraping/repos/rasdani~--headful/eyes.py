import os
from natsort import natsorted
from dotenv import load_dotenv
import base64
import requests
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

LABEL_DESCRIPTION = "small yellow"
# LABEL_DESCRIPTION = "large red"
# LABEL_DESCRIPTION = "small red"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6))
def see(image_path, user_request):
    base64_image = encode_image(image_path)
    message = """\
This is a screenshot of a web page. The clickable elements of the page have red bounding boxes are annotated with {label_descripton} labels containing a code of one or two letters.\n
Now follows a user request. Indentify the element the user wants to interact with and return the corresponding letter code.\
Respond by citing the letter code only, or respond with a question mark, if the requested element is not present in the screenshot!\n
USER REQUEST: {user_request}"""
    message = message.format(
        user_request=user_request, label_descripton=LABEL_DESCRIPTION
    )

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": "gpt-4-vision-preview",
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    # {"type": "text", "text": messages},
                    {
                        "type": "image_url",
                        "image_url": {
                            # "url": f"data:image/jpeg;base64,{base64_image}"
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    response_json = response.json()
    if 'error' in response_json:
        print(f"Error: {response_json['error']['message']}")
        return None
    response_message = response_json["choices"][0]["message"]["content"]
    print(response_message)
    return response_message


def see_legacy(image_paths, user_request):
    base64Frames = [encode_image(image_path) for image_path in image_paths]
    #     message = """\
    # ## Description
    # These are two screenshots of the same webpage. The first one just shows the webpage, the second one overlays the webpage with {label_description} labels. \
    # All clickable elements of the page are annotated with these {label_description} labels containing one or two letters. \
    # ## Objective
    # Now follows a request. Indentify the relevant webpage element by looking at the first screenshot and look up the corresponding annotation label by referencing the second screenshot. \
    # Respond by citing the letter code only!\
    # ## Request
    # {user_request}"""
    #     message = message.format(
    #         user_request=user_request, label_description=LABEL_DESCRIPTION
    #     )

    message = """\
This is a screenshot of a web page. One clickable element of the web page is highlighted with a red bounding box.\n
Now follows a user request. Classify wether the user request relates to the highlighted web page element or not.\
Respond with a simple 'yes' or 'no'. When you are uncertain still respond with 'no', don't converse with the user, don't explain, avoid superfluos chatter.\n
USER REQUEST: {user_request}"""
    message = message.format(user_request=user_request)
    # messages = [message] * len(base64Frames)
    # PROMPT_MESSAGES = [
    #     {
    #         "role": "user",
    #         "content": [
    #             message,
    #             # *map(lambda x: {"image": x, "resize": 768}, base64Frames),
    #             *map(lambda x: {"image": x, "detail": "high"}, base64Frames),
    #         ],
    #     },
    # ]
    PROMPT_MESSAGES = []
    for image in base64Frames:
        # for image_path in image_paths:
        # print("Processing image:", image_path)
        prompt = {
            "role": "user",
            "content": [
                {"type": "text", "text": message},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}"
                        # "detail": "high",
                    },
                },
            ],
        }
        PROMPT_MESSAGES.append(prompt)

    # params = {
    #     "model": "gpt-4-vision-preview",
    #     "temperature": 0.0,
    #     # "messages": PROMPT_MESSAGES,
    #     "prompt": PROMPT_MESSAGES,
    #     # "api_key": api_key,
    #     # "headers": {"Openai-Version": "2020-11-07"},
    #     "max_tokens": 200,
    # }

    # # result = openai.ChatCompletion.create(**params)
    # result = client.chat.completions.create(**params)
    # response = result.choices[0].message.content

    # no batching for chat completion :(
    for i, prompt in enumerate(PROMPT_MESSAGES):
        params = {
            "model": "gpt-4-vision-preview",
            "temperature": 0.0,
            "messages": [prompt],
            # "prompt": PROMPT_MESSAGES,
            # "api_key": api_key,
            # "headers": {"Openai-Version": "2020-11-07"},
            "max_tokens": 200,
        }

        result = client.chat.completions.create(**params)
        response = result.choices[0].message.content
        print(f"{i}: {response}")
    # breakpoint()
    return response


if __name__ == "__main__":
    # image_path = "screenshot_after.png"
    image_files = [
        os.path.join("bbox-images/", f)
        for f in os.listdir("bbox-images/")
        if f.endswith(".png")
    ]
    image_files = natsorted(image_files)
    # API rate limits
    image_files = image_files[:20]
    # image_files = image_files[:2]
    print(image_files)
    while True:
        user_request = input("User request: ")
        see_legacy(image_files, user_request)

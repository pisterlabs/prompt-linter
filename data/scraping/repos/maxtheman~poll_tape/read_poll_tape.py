# %%
from openai import OpenAI
import pandas as pd
from typing import Tuple, List, Dict, Optional
from pydantic import BaseModel
import base64
import glob
import os
import json
from pprint import pprint
from tqdm import tqdm
from IPython.display import Image 

# %%
def load_image_uris(directory: str = '../data') -> list:
    """
    This function loads all of the image URIs recursively from a given directory.
    There may be subdirectories. Output is the locations at which they can be accessed 
    to load into memory one at a time.
    
    Args:
        directory (str): The directory from which to load the image URIs. Default is '../data'.
    
    Returns:
        list: A list of image URIs.
    """
    image_uris = glob.glob(directory + '/**/*.jpg', recursive=True)
    return image_uris

image_uris = load_image_uris()
pprint(image_uris[0:10])
len(image_uris)

# %% [markdown]
# Pydantic schema below allows you to define the schema up-front and then parse it into pandas later, rejecting or handing invalid data.
# 
# None represents unreadable data.

# %%
class Result(BaseModel):
    Candidate: Optional[str] = None
    Votes: Optional[str] = None

class Contest(BaseModel):
    Office: Optional[str] = None
    Results: List[Result]

class PollTape(BaseModel):
    County: Optional[str] = None
    Precinct: Optional[str] = None
    Contest: List[Contest]

pprint(PollTape.model_json_schema())


# %% [markdown]
# Here's the tool schema. Unfortunately we can't use this or JSON mode while GPT-4V is in preview, but it's here for when we can.
# 
# docs:
# 
# https://platform.openai.com/docs/guides/vision
# 
# https://platform.openai.com/docs/guides/text-generation/json-mode

# %%
tools = [
    {
        "type": "function",
        "function": {
            "name": "process_poll_tape",
            "description": "Process a poll tape from an election. Some parameters may be null where the poll tape is not legible, as indicated in description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "PollTape": {
                        "type": "object",
                        "description": "The poll tape object to be processed",
                        "properties": {
                            "County": {"type": "string", "description": "The county of the poll tape. Provide null if not legible."},
                            "Precinct": {"type": "string", "description": "The precinct of the poll tape. Provide null if not legible."},
                            "Contest": {
                                "type": "array",
                                "description": "The list of contests in the poll tape",
                                "items": {
                                    "type": "object",
                                    "description": "A contest in the poll tape",
                                    "properties": {
                                        "voter_option": {"type": "string", "description": "The voter option of the contest (e.g. candidate name). Provide null if not legible."},
                                        "votes": {"type": "integer", "description": "The number of votes for the voter option. Provide bull if not legible."}
                                    },
                                    "required": ["voter_option", "votes"]
                                }
                            }
                        },
                        "required": ["County", "Precinct", "Contest"]
                    }
                },
                "required": ["PollTape"],
            },
        },
    }
]

# %%
SYSTEM_PROMPT = """The images you recieve are of poll tapes from elections. Your goal is to output JSON.
Because they are election images, it essential that the results you provide are correct. Do not guess, provide null if the data is not available. Do not output data in an incorrect schema, including pre-pending with ```json or appending with ```. Doing so will harm the users of this data.
Read the poll tape and return the JSON in the below schema.`.
Here is the schema for the poll tape:{0}
Remember: the output must be the valid JSON or you risk causing user harm.""".format(PollTape.model_json_schema())

# %%
# client = OpenAI(api_key="YOUR_API_KEY_HERE" 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_transcription(encoded_image):
    """
    This function takes an encoded image and returns the transcription of the image.
    The transcription is obtained by using the OpenAI GPT-4 Vision model.

    Unfortunately, The model doesn't take the `tools` or `json_object` response formats.

    Parameters:
    encoded_image (str): The base64 encoded string of the image.

    Returns:
    dict: The response from the OpenAI API containing the transcription of the image.
    """
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        # response_format={ "type": "json_object" },
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    },
                ]
            }
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content

# %% [markdown]
# Run once by hand to test

# %%
test_image = image_uris[0]
with open(test_image, 'rb') as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
response = get_transcription(encoded_string)

# %%
Image(filename=test_image)


# %%
print(response)

# %%
pprint(PollTape(**json.loads(response)).dict())


# %%
def process_images(image_uris) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    This function takes a list of image URIs and returns a dataframe of the poll tape data
    and a list of non-compliant schemas.
    """
    df_list = []
    non_compliant_schemas = []

    for image_uri in tqdm(image_uris):
        try:
            encoded_string = None
            with open(image_uri, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            if encoded_string is None:
                raise Exception("Could not encode image")
            response = get_transcription(encoded_string)
            poll_tape = PollTape(**json.loads(response))
            for contest in poll_tape.Contest:
                for result in contest.Results:
                    row = {
                        'County': poll_tape.County,
                        'Precinct': poll_tape.Precinct,
                        'Office': contest.Office,
                        'Candidate': result.Candidate,
                        'Votes': result.Votes
                    }
                    df_list.append(pd.DataFrame(row, index=[0]))
        except Exception as e:
            non_compliant_schemas.append({
                'image_uri': image_uri,
                'error': str(e),
                'response': response,
            })

    df = pd.concat(df_list, ignore_index=True)
    return df, non_compliant_schemas

# %% [markdown]
# This may take some time to run. Response from the API is anywhere from 4-20 seconds.
# 
# Uses tqdm to do the timing so you'll know how long it takes to run.

# %%
number_to_process = 10
df, non_compliant_schemas = process_images(image_uris[0:number_to_process])
df.to_csv('processed_images.csv', index=False)
pprint(non_compliant_schemas)

# %%
df.sample(15)


# %%
pprint(non_compliant_schemas)
Image(filename=non_compliant_schemas[1]['image_uri'])

# %% [markdown]
# Below code is for command line usage only

# %%
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Process some images.')

# Add the arguments
parser.add_argument('directory', metavar='directory', type=str, help='the directory to process images from')
parser.add_argument('num_images', metavar='num_images', type=int, help='the number of images to process')

# Parse the arguments
args = parser.parse_args()

# Now you can use args.directory and args.num_images in your script
image_uris = load_image_uris(args.directory)
df, non_compliant_schemas = process_images(image_uris[0:args.num_images])
df.to_csv('processed_images.csv', index=False)

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from openai import OpenAI
import os
import base64
import json

client = OpenAI()

with open("./01.html", "rb") as html_file:
    html01 = html_file.read()
with open("./02.html", "rb") as html_file:
    html02 = html_file.read()
with open("./03.html", "rb") as html_file:
    html03 = html_file.read()

with open("./image.png", "rb") as image_file:
    base64_utf8_str = base64.b64encode(image_file.read()).decode("utf-8")
    image01 = f"data:image/png;base64,{base64_utf8_str}"
with open("./image02.png", "rb") as image_file:
    base64_utf8_str = base64.b64encode(image_file.read()).decode("utf-8")
    image02 = f"data:image/png;base64,{base64_utf8_str}"
with open("./image03.jpeg", "rb") as image_file:
    base64_utf8_str = base64.b64encode(image_file.read()).decode("utf-8")
    image03 = f"data:image/jpeg;base64,{base64_utf8_str}"

prompt01 = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": f"{image01}"},
            {
                "type": "text",
                "text": f"""
                The product is a web application that is essentially a CMS for 3d
                product configurators. You can upload 3d models, textures, and
             environments. You can create materials, and then you can combine them
             together in what we call a 'konfig', which allows you to add options,
             combine the options into steps, add pricing, SKU's, etc.
             
             I want to create a demo that shows how to upload an environment.

             Here is the html: {html01}
             """,
            },
        ],
    },
]

prompt02 = [
    {
        "role": "system",
        "content": """
        ```json
            {
              "description": "This page appears to be the main dashboard of a 3D product configurator CMS where users can upload and manage different assets such as materials, textures, meshes, previews, and environments for 3D configurations.",
              "guess": {
                "01": {
                  "id": "e",
                  "action": "Navigates to the Environments section where users can manage and upload new environment assets."
                },
                "02": {
                  "id": "9",
                  "action": "Initiates the creation process, likely to be used here for adding a new environment asset."
                },
                "03": {
                  "id": "0",
                  "action": "Takes the user to the Konfigs section, potentially for managing or creating a new 'konfig' that could include the environment."
                }
              }
            }
         ```
         """,
    },
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": f"{ image02 }"},
            {
                "type": "text",
                "text": f"""
                             I clicked on id "60". That took me to this page, So we followed this action "Takes the user to the section where they can manage environments. This is likely the first place they would go to in order to start the process of uploading an environment as per the demo's objective.".

                             In a demo with this initial prompt:

                                "The product is a web application that is essentially a CMS for
                                3d product configurators. You can upload 3d models, textures,
                                and environments. You can create materials, and then you can
                                combine them together in what we call a 'konfig', which allows
                                you to add options, combine the options into steps, add
                                pricing, SKU's, etc.
            It is possible elements exist in the HTML that don't exist in the
                                I want to create a demo that shows how to upload an environment."
 
                             What's next?

                             Here is the html: {html02}
                             """,
            },
        ],
    },
]
prompt03 = [
    {
        "role": "system",
        "content": """
        ```json
            {
              "description": "Since the environments section has been accessed, the next action in the demo should logically be to show how an environment is uploaded to the CMS as per the new objective.",
              "guess": {
                "01": {
                  "id": "8",
                  "action": "Opens the upload dialog or interface for adding a new environment asset."
                },
                "02": {
                  "id": "1c",
                  "action": "An alternate click target which is part of the upload button, possibly invoking the same upload action."
                },
                "03": {
                  "id": "b",
                  "action": "Another element within the upload button that could be interacted with to start the upload process."
                }
              }
            }
            ```
         """,
    },
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": f"{ image03 }"},
            {
                "type": "text",
                "text": f"""
                             I clicked on id "8". That took me to this page, So we followed this action "Opens the upload dialog or interface for adding a new environment asset.".

                             In a demo with this initial prompt:

                                "The product is a web application that is essentially a CMS for
                                3d product configurators. You can upload 3d models, textures,
                                and environments. You can create materials, and then you can
                                combine them together in what we call a 'konfig', which allows
                                you to add options, combine the options into steps, add
                                pricing, SKU's, etc.
                                I want to create a demo that shows how to upload an environment."
 
                             Where should the user click next?

                             Here is the html: {html03}
                             """,
            },
        ],
    },
]

# ```json
# {
  # "description": "This is the upload interface for environments within the CMS for 3D product configurators. The user needs to complete the action of uploading an environment file to the system.",
  # "guess": {
    # "01": {
      # "id": "05",
      # "action": "Clicking on this would likely initiate the actual upload process after files have been selected or dragged into the upload area."
    # },
    # "02": {
      # "id": "6",
      # "action": "This might open the file explorer to select the .hdr file to upload."
    # },
    # "03": {
      # "id": "1",
      # "action": "The user may click here to possibly view information or instructions about uploading files, or it might be a part of the overall upload component."
    # }
  # }
# }
# ```

completion = client.chat.completions.create(
    model="gpt-4-vision-preview",
    max_tokens=4096,
    messages=[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """
            You are a sales consultant / solutions engineer. You help companies
            create demos of their products. Their products are software
            applications that are built using web technologies. You are going
            to help someone make a demo. You will be presented with a prompt
            that holds information about the product the client wants to
            present, about their prospect, and a rough description on what the
            product does. You will then receive a screenshot of the page, with
            the corresponding HTML structure. It's simplified. We
            only have the tags, and every element has it's
            own id, which is inside of the html's 'i' tag. I want you to think
            of 3 places you'd possibly click next, given our customers prompt.
            Then I want you to match each of them to a place in the HTML
            structure. The response we want is a JSON object, with the
            following schema (psuedo-code)

            ``` 
                type guess = { 
                    id: <id-to-click>,
                    action: <what would happen when you click there>
                }; 
                type guesses = {01: guess, 02: guess, 03: guess} 
                type response = {
                    description: <summary of what this page is for>, 
                    guess: guesses, 
                }
            ```

            - Return only the 3 guesses in the correct format
            """,
                }
            ],
        },
    ]
    + prompt03,
)
print(completion.choices[0].message.content)

"""
| Badger Go-Wild Utils.
| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import base64
import os
import re

import openai


RANDOM_SVG_PROMPT = """
Generate a detailed, high-quality SVG logo of a SUBJECT with proper XML 
formatting. Make sure to include the XML header, and keep the design simple but 
visually appealing. The SVG should be suitable for use as a badge logo. Include 
some details in the design, so that it stands out from other badges. Only return
the SVG data and XML header, not any explanatory text. The badge should NOT have
any text in it, as that will be added later.
"""

FIX_BADGE_PROMPT = """
Turn the text below into a working SVG code for SUBJECT. The SVG code should be 
properly formatted and include the XML header. The badge should NOT have any 
text in it, as that will be added later. Just return the SVG data and XML 
header, not any explanatory text. You will be penalized if the SVG code is not
properly formatted, or if it contains any text. The text to correct is:\n\n
"""


def generate_random_svg(subject):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("Calling the OpenAI API... (this may take a few seconds)")
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": RANDOM_SVG_PROMPT.replace("SUBJECT", subject),
            },
        ],
    )
    response = completion.choices[0].message["content"]

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": FIX_BADGE_PROMPT.replace("SUBJECT", subject)
                + response,
            },
        ],
    )
    response = completion.choices[0].message["content"]

    svg = response.replace('\\"', '"').replace("\\n", "\n").strip()
    svg = re.search(r"<svg.*?</svg>", svg, re.DOTALL)
    if svg:
        return svg.group(0).encode("utf-8")
    else:
        print("GPT-3.5 failed to generate a valid SVG. Please try again.")
        return


def generate_trial_badge(svg_data, subject):
    # Encode in base64 and decode it to ASCII
    b64_logo = base64.b64encode(svg_data).decode("ascii")

    subject = subject.replace(" ", "%20").strip()

    badge_pattern = (
        f"[![Trial Badge](https://img.shields.io/badge/{subject}-blue.svg?"
        f"style=flat&logo=data:image/svg+xml;base64,{b64_logo})](https://github.com/voxel51/badger)"
    )
    return badge_pattern

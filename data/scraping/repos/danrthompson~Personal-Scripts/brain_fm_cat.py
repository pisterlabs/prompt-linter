# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os

import cv2
import pytesseract
import openai

from openai_utils.src.openai_utils import (
    set_openai_api_key,
    get_text_from_openai_chat_completion,
    get_openai_chat_completion,
)

# %%
# Set openai api key with helper
set_openai_api_key(openai)
# Path to Tesseract executable (replace with your own path)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


# %%
PROMPT = """
I have taken a screenshot of a description of a song that I'm listening to. I already used OCR to parse the text from the image. Now I would like you to categorize that text into a particular format.

I am trying to capture the following information:
| Mental state | Activity | Genre | Track title | Neural effect level | Musical complexity | Mood | Instrumentation |

Here is a sample input from a previous screenshot:
=====BEGIN SAMPLE=====
7 = full capacity

a. ELECTRONIC - HIGH NEURAL EFFECT

Sw ODA

track Information similar tracks

mental state activity
focus deep work

musical complexity neural effect level

= medium = high

mood
chill - upbeat

instrumentation
electronic percussion + arp synth - arp synth bass
=====END SAMPLE=====

Here is the correct output for the sample input:
=====BEGIN OUTPUT=====
| Focus | Deep work | Electronic | Full capacity | High | Medium | Chill - upbeat | Electronic percussion - Arp synth - Arp synth bass |
=====END OUTPUT=====

Please categorize the following text into the correct format. You can use the sample input as a guide. If you have any questions, please ask. Thank you!

=====BEGIN INPUT=====
{input}
=====END INPUT=====
"""


# %%
def get_text_from_screenshot(filename: str) -> str:
    image = cv2.imread(os.path.expanduser(image_path))

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to convert grayscale image to binary image
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Apply dilation and erosion to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_image = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    # Use Tesseract to extract text from image
    text: str = pytesseract.image_to_string(clean_image)

    # Return extracted text
    return text


# %%
image_path = "data/computer_text.jpg"


# %%
input_text = get_text_from_screenshot(image_path)
prompt_with_input = PROMPT.format(input=input_text)

# %%
completion: OpenAIObject = get_openai_chat_completion(prompt_with_input)
# %%
completion
# %%

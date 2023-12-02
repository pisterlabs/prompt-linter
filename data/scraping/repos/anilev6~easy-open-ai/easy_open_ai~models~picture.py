import openai

import os
import base64

from .api_key import OPENAI_API_KEY

from .error_handling import handle_openai_error, ahandle_openai_error


def normalize_file_name(file_name):
    # Remove leading and trailing whitespace
    file_name = file_name.strip()

    # Replace spaces with underscores
    file_name = file_name.replace(" ", "_")

    # Remove any remaining characters that are not alphanumeric, underscores, or periods
    file_name = "".join(c for c in file_name if c.isalnum() or c in ("_",))

    # Limit the length of the file name (adjust as needed)
    max_length = 255  # Typical limit for file names on many file systems
    file_name = file_name[:max_length]

    return file_name


def pic_saver(b64_json_string, file_name):  # no format
    # Decode the base64 encoded string
    decoded_json_string = base64.b64decode(b64_json_string)
    # Normalize the file name
    normalized_file_name = normalize_file_name(file_name)
    file_path = os.path.join(os.getcwd(), f"{normalized_file_name}.png")
    with open(file_path, "wb") as f:
        f.write(decoded_json_string)


# ------------------------------------------------------Generate Pictures from Text------------------------------------------------------------


class PictureGenerator:
    """
    Generated images can have a size of 256x256, 512x512, or 1024x1024 pixels.
    The more detailed the description, the more likely you are to get the result that you or your end user want.
    Each image can be returned as either a URL or Base64 data, using the response_format parameter.
    1<=n<=10;
    response_format: one of url or b64_json; URLs will expire after an hour.
    """

    def __init__(self, user_input: str, n=1, format="url"):
        self.user_input = user_input
        self.n = n  # length of the output list; amount of options
        self.format = format  #'b64_json'

    @handle_openai_error
    def task(self) -> list:
        # raise openai.error.APIError # test error handling https://github.com/openai/openai-python/blob/main/openai/error.py
        openai.api_key = OPENAI_API_KEY
        response = openai.Image.create(
            prompt=self.user_input, n=self.n, response_format=self.format
        )
        result = []
        for i in response["data"]:
            result.append(i[self.format])
        return result

    @ahandle_openai_error
    async def async_task(self) -> list:
        openai.api_key = OPENAI_API_KEY
        response = await openai.Image.acreate(
            prompt=self.user_input, n=self.n, response_format=self.format
        )
        result = []
        for i in response["data"]:
            result.append(i[self.format])
        return result

import os
import requests
from openai_util.function_call.openaifunc_decorator import openai_func


@openai_func
def download_file_from_url(url: str, save_filename: str):
    """
    download file from a url and save it to a file
    :param url: the url of the file
    :param save_filename: the filename to save
    """
    # Make sure the tmp folder exists
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # Save the file under the tmp folder
    local_filename = os.path.join('tmp', save_filename)

    # Send an HTTP request to URL
    response = requests.get(url, stream=True)

    # Make sure the request is successful
    response.raise_for_status()

    # open the file for writing
    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            # If there is a chunk, write to the file
            if chunk:
                f.write(chunk)

    return os.path.abspath(local_filename)

# print(download_file(
#     'https://ask.qcloudimg.com/http-save/yehe-6781431/fc477e39f845d3a223e42da74bbf645a.png?imageView2/2/w/1200',
#     'demo.jpg'))

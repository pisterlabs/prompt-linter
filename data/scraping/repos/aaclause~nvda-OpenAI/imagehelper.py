import base64
import os
import sys
import re
from logHandler import log
from .consts import ADDON_DIR

additionalLibsPath = os.path.join(ADDON_DIR, "lib")
sys.path.insert(0, additionalLibsPath)
from openai import OpenAI
sys.path.remove(additionalLibsPath)

# Borrowed from <https://platform.openai.com/docs/guides/vision>

# Function to encode the image
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')


def describeFromImageFileList(
	client,
	pathList,
	prompt=None,
	max_tokens=700
):
	"""
	Describe a list of images from a list of file paths.
	@param client: OpenAI client
	@param pathList: list of file paths
	@param prompt: prompt to use
	@param max_tokens: max tokens to use
	@return: description
	"""
	if not prompt or not prompt.strip():
		return None
	content = [
		{
			"type": "text",
			"text": prompt,
		}
	]
	for path in pathList:
		url_re = re.compile(r"^https?://")
		if url_re.match(path):
			content.append(
				{
					"type": "image_url",
					"image_url": {
						"url": path,
					},
				}
			)
		elif os.path.isfile(path):
			base64_image = encode_image(path)
			format = path.split(".")[-1]
			mime_type = f"image/{format}"
			content.append(
				{
					"type": "image_url",
					"image_url": {
						"url": f"data:{mime_type};base64,{base64_image}"
					},
				}
			)
		else:
			raise ValueError("Invalid path: {}".format(path))
	response = client.chat.completions.create(
		model="gpt-4-vision-preview",
		messages=[
			{
				"role": "user",
				"content": content,
			}
		],
		max_tokens=max_tokens
	)
	return response.choices[0]


import openai

"""response = openai.Image.create(
  prompt="Undertale fight against Sans, but Hamilton pixel art is there instead of Sans.",
  n=1,
  size="256x256"
)
print(response)
image_url = response['data'][0]['url']
print(image_url)"""

response = openai.Image.create_edit(
  image=open("SansFight.png", "rb"),
  mask=open("SansFightMask.png", "rb"),
  prompt="Sans fight GUI, but Hamilton replaces Sans and is dressed like Sans.",
  n=1,
  size="512x512"
)
print(response)
print(response['data'][0]['url'])

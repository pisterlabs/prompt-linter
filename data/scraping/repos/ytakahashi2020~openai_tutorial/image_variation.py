from openai import OpenAI
client = OpenAI()

response = client.images.create_variation(
    image=open("./image_edit_original.png", "rb"),
    n=1,
    size="1024x1024"
)

image_url = response.data[0].url

print(response)

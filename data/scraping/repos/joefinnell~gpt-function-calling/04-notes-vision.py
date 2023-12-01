from openai import OpenAI
import base64

# Set the OpenAI API key
client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "images/notes.png"

# Getting the base64 string
base64_image = encode_image(image_path)
image_url = f"data:image/jpeg;base64,{base64_image}"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "what are the main pieces of information from the notes in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        ]
    }
]

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=messages,
)

print(response.choices[0].message.content)

import os
import argparse
import base64
from openai import OpenAI

# Initialize the OpenAI client and set the API key
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

# Model name
model = "gpt-4-vision-preview"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    # Create an ArgumentParser to handle command-line arguments
    parser = argparse.ArgumentParser(description="Generate a description for an image.")
    parser.add_argument("image_path", help="Path to the image file")

    args = parser.parse_args()

    # Getting the base64 string from the image file
    base64_image = encode_image(args.image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    message_content = response.choices[0].message.content
    print(message_content)

if __name__ == "__main__":
    main()

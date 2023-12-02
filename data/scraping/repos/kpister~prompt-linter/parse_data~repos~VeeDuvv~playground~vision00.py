from openai import OpenAI

import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize the OpenAI client
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is happening in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS-q64uKF5xW1LAzzHwtRIZv9NBGBo0zGOJfw&usqp=CAU",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])
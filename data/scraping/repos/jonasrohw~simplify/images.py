import openai
import requests

# Set up OpenAI API key
openai.api_key = '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'

def fetch_image_from_openai(prompt):
    # Use OpenAI to get an image URL based on the prompt
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']

    # Download the image using the URL
    image_data = requests.get(image_url).content

    # Save the image to a local file
    with open('output_image.jpg', 'wb') as image_file:
        image_file.write(image_data)

    print(f"Image saved to output_image.jpg")

if __name__ == "__main__":
    prompt = input("Enter the image prompt: ")
    fetch_image_from_openai(prompt)

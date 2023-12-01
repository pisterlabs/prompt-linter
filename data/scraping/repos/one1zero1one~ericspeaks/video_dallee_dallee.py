import os
import openai
import requests

# Assuming the OpenAI client is already authenticated
client = openai.OpenAI()

def extract_prompts_from_srt(file_path):
    """Extract prompts from the SRT file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    prompts = [lines[i+2].strip() for i in range(0, len(lines), 4) if i + 1 < len(lines)]
    return prompts

def generate_image(prompt, image_index, output_folder):
    """Generate an image using DALL-E API and save it."""
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    image_url = response.data[0].url

    # Fetch and save the image
    output_path = os.path.join(output_folder, f"{image_index}.jpg")
    image_response = requests.get(image_url)
    with open(output_path, "wb") as file:
        file.write(image_response.content)

def main():
    input_file_path = "sources/eric1_image_prompts.srt"
    output_folder = "sources/eric1_images"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    prompts = extract_prompts_from_srt(input_file_path)

    for index, prompt in enumerate(prompts):
        print(f"Generating image for prompt {index}: {prompt}")
        generate_image(prompt, index, output_folder)

    print("All images have been generated.")

if __name__ == "__main__":
    main()

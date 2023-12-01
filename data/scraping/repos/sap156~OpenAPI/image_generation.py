import openai
import os

openai.api_key = 'your-api-key'  # replace 'your-api-key' with your actual API key

try:
    response = openai.Completion.create(engine="text-davinci-003", prompt="Test", max_tokens=5)
    print("API key is valid.")
except Exception as e:
    print("API key is invalid. Error: ", str(e))

    
def generate_image_from_text():
    prompt = input("Enter a prompt for the image: ")
    n = int(input("Enter the number of images to generate (1-10): "))
    size = input("Enter the size of the image (256x256, 512x512, or 1024x1024): ")
    
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=n,
            size=size
        )
        image_url = response['data'][0]['url']
        print(image_url)
    except Exception as e:
        print(f'Error occurred while generating image: {e}')

def modify_image():
    image_path = input("Enter the path to the image file: ")
    mask_path = input("Enter the path to the mask file: ")
    prompt = input("Enter a prompt for the new image: ")
    n = int(input("Enter the number of images to generate (1-10): "))
    size = input("Enter the size of the image (256x256, 512x512, or 1024x1024): ")

    try:
        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            response = openai.Image.create_edit(
                image=image_file,
                mask=mask_file,
                prompt=prompt,
                n=n,
                size=size
            )
        image_url = response['data'][0]['url']
        print(image_url)
    except Exception as e:
        print(f'Error occurred while modifying image: {e}')

def generate_image_variation():
    image_path = input("Enter the path to the image file: ")
    n = int(input("Enter the number of variations to generate (1-10): "))
    size = input("Enter the size of the image (256x256, 512x512, or 1024x1024): ")

    try:
        with open(image_path, "rb") as image_file:
            response = openai.Image.create_variation(
                image=image_file,
                n=n,
                size=size
            )
        image_url = response['data'][0]['url']
        print(image_url)
    except Exception as e:
        print(f'Error occurred while generating image variation: {e}')

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Generate image from text")
    print("2. Modify an existing image")
    print("3. Generate a variation of an existing image")
    choice = int(input("Enter your choice (1-3): "))

    if choice == 1:
        generate_image_from_text()
    elif choice == 2:
        modify_image()
    elif choice == 3:
        generate_image_variation()
    else:
        print("Invalid choice. Please enter a number between 1 and 3.")
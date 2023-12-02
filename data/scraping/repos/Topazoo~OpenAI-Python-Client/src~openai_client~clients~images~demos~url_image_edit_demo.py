from ..clients import URL_Image_Edit_Client

# Simple image edit app :)
if __name__ == "__main__":
    # You can open the file yourself
    image = open("src/openai_client/clients/images/demos/image.png", "rb")
    # Or just pass a string path
    mask = "src/openai_client/clients/images/demos/mask.png"

    # API Key is read from OPENAI_API_KEY
    client = URL_Image_Edit_Client(image, mask)

    # Prompt the user for input (e.g. "a goofy looking cartoon smiley face")
    face = input("Describe a face to generate:\n>>> ")

    # Send the request and get the image URL
    image_url = client.run_prompt(face)

    # Get the image URL
    print(image_url)

from ..clients import URL_Image_Variation_Client

# Simple image variation app :)
if __name__ == "__main__":
    image = open("src/openai_client/clients/images/demos/image.png", "rb")

    # API Key is read from OPENAI_API_KEY
    client = URL_Image_Variation_Client(image)

    # Send the request and get the image URL
    image_url = client.run_prompt()

    # Get the image URL
    print(image_url)

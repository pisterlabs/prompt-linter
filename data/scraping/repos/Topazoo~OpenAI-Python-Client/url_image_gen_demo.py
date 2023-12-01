from ..clients import URL_Image_Client

# Simple animal mashup app :)
if __name__ == "__main__":
    # API Key is read from OPENAI_API_KEY
    client = URL_Image_Client()

    # Add a context to always include before the prompt that is sent to the API
    client.add_pre_prompt_context("Generate a hybrid animal using the following animals:")
    # Add a context to always include after the prompt that is sent to the API
    client.add_post_prompt_context("This rendering should be hyperrealistic. The background \
                                   should be a savannah during the daytime")

    # Prompt the user for input
    animals = input("Choose two animals to create a hybrid of:\n>>> ")

    # Send the request and get the image URL
    image_url = client.run_prompt(animals)

    # Get the image URL
    print(image_url)
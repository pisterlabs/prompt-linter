"""dalleImage class."""
import openai

# A class for generating images using the OpenAI DALL·E model.
class dalleImage:
    def __init__(self, api_key):
        """
        Initialize the DALL·E Image Generator class

        Args:
            api_key (str): Your OpenAI API key for authentication.

        Constructor to set up the OpenAI API key for authentication.
        """
        openai.api_key = api_key

    def generate_image(self, prompt, n=2, size="256x256"):
        """
        Generate an image based on a given prompt.

        Args:
            prompt (str): Prompt for image generation.
            n (int, optional): Number of image alternatives to generate. Default is 2.
            size (str, optional): Size of the generated image in the format "widthxheight". Default is "256x256".

        Returns:
            dict: Generated image response from the OpenAI DALL·E model.

        This method calls the OpenAI DALL·E API to generate images based on the provided prompt
        and other parameters. It returns a dictionary containing the image response.
        """
        # Create the image using the OpenAI API
        response = openai.Image.create(
            prompt=prompt,
            n=n,
            size=size
        )
        return response

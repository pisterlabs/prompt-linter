from langchain.tools import BaseTool
from dotenv import load_dotenv
import os
import requests
import io
from PIL import Image
from pathlib import Path

load_dotenv()


# TODO: store the image on cloud storage and return the URL with SAS token
# TODO: add json output parser
# TODO: add prompt template
# ref: https://pub.aimind.so/building-a-custom-chat-agent-for-document-qa-with-langchain-gpt-3-5-and-pinecone-e3ae7f74e5e8
class ImageGeneratorTool(BaseTool):
    """
    Tool to create an image from text using the HuggingFace inference endpoint.
    Use this tool only when the user asks to create an image from text.

    """

    name = "Image Generator"
    description = """
    Use this tool ONLY WHEN user ask to GENERATE an image from text. 
    Do not use this tool for other purposes. 
    ONLY Keywords for using this tool are : 'generate' and 'image'.
    Keywords for NOT using this tool are : 'tell', 'provide', 'write'
    """

    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

    def _call_hf_inference_api(self, payload):
        """Call the HuggingFace API and return the response content."""
        response = requests.post(url=self.API_URL, headers=self.headers, json=payload)
        return response.content

    def _get_path(self, query: str) -> Path:
        """Generate a path for the image based on the provided query."""
        local_name = f"{query.replace(' ', '_')}.png"
        user_folder = os.getenv("IMAGES_USER_FOLDER")
        if not user_folder:
            raise EnvironmentError(
                "IMAGES_USER_FOLDER environment variable is not set."
            )
        path = Path(user_folder)
        return path / local_name

    def _generate_image_from_bytes(self, image_bytes) -> Image:
        """Convert bytes to an Image object."""
        return Image.open(io.BytesIO(image_bytes))

    def _generate_save_and_return_image(self, query: str) -> str:
        """Generate, save, and return a message about the saved image."""
        image_bytes = self._call_hf_inference_api({"inputs": query})
        image = self._generate_image_from_bytes(image_bytes)

        image_path = self._get_path(query=query)
        image.save(image_path)

        return (
            f"Image generated successfully and saved in the user folder. "
            f"The image is available here: {image_path}"
        )

    def run(self, query: str) -> str:
        return self._generate_save_and_return_image(query=query)

    def _run(self, query: str) -> str:
        return self._generate_save_and_return_image(query=query)

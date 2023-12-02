from openai import OpenAI


class DALLE:

    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.images.generate(prompt=prompt,
                                               model="dall-e-3",
                                               n=1,
                                               quality="standard",
                                               size="1024x1024")
        return response.data[0].url


if __name__ == "__main__":
    # arguemnts
    api_key = "___API_KEY___"
    prompt = "___PROMPT_TO_GENERATE_IMAGE___"

    # chat
    dalle = DALLE(api_key=api_key)
    image_url = dalle.generate(prompt=prompt)
    print(f"image: {image_url}")

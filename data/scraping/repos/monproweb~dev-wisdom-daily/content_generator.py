import re
from openai import OpenAI
from mongo_manager import insert_quote, get_last_50_quotes

client = OpenAI()


class ContentGenerator:
    def __init__(self, config):
        self.config = config

    def generate_quote(self):
        previous_quotes = get_last_50_quotes()
        chat_messages = [
            {
                "role": "assistant",
                "content": "Here are the last 50 quotes that have been generated: "
                + "\n".join(previous_quotes)
                + ". Please generate a new quote that is different from these previous ones, and ensure equal representation of all domains.",
            },
            {
                "role": "user",
                "content": "Share a thought-provoking and concise quote that captures the spirit of a specific domain within the tech industry, such as artificial intelligence, web3 development, software development, game development, cybersecurity, or data science. The quote should come from a respected figure in the specified domain and resonate within the tech community. Use 1-2 relevant hashtags and emojis to enhance engagement. Present the quote first, followed by the individual's name and emojis, and end with the appropriate hashtags.",
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=chat_messages,
            n=1,
            stop=None,
            temperature=0.7,
            max_tokens=60,
        )

        if response.choices:
            quote = response.choices[0].message.content.strip()
            quote_text = re.search(r'"(.*?)"', quote).group(1)
            insert_quote(quote_text)
            return quote, quote_text

        return "", ""

    def generate_detailed_description(self, quote_text):
        chat_messages = [
            {
                "role": "user",
                "content": f"Imagine a vivid and intricate visual representation that is deeply inspired by the quote \"{quote_text}\". This image should be rich in detail, allowing for a comprehensive exploration of the quote's meaning and emotion. Incorporate various adjectives, locations, or artistic styles to enhance the image's depth and appeal. Additionally, consider infusing the image with surreal or fantastical elements to make it even more captivating. Your goal is to create a scene or metaphor that is both imaginative and engaging, drawing the viewer into a world that reflects the essence of the quote.",
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=chat_messages,
            n=1,
            stop=None,
            temperature=0.8,
            max_tokens=150,
        )

        detailed_description = response.choices[0].message.content.strip()
        return detailed_description

    def generate_image(self, prompt):
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url
        return image_url

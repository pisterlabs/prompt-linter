import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# OpenAI API key
client = OpenAI(api_key=os.getenv('api_key'))


def get_recipes(seasonings: [str], items: [str]):
    # Create a conversation-like prompt based on the input
    prompt_text = f"Generate one recipe based on the following seasonings and items:\nSeasonings: {', '.join(seasonings)}\nItems: {', '.join(items)}"
    
    # Call the OpenAI API with the prompt
    response = client.chat.completions.create(model="gpt-4",  # Or the most appropriate model you have access to
    messages=[
        {"role": "system", "content": f"You are a helpful assistant providing recipes."},
        {"role": "user", "content": prompt_text}
    ])

    # Extract the response
    message_content = response.choices[0].message.content
    return message_content


def get_image(answer: str):
    """Get an image from the OpenAI API based on the answer to the prompt"""
    try: 
        image_response = client.images.generate(
            model="dall-e-3",
            prompt=answer,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return image_response.data[0].url
    except Exception as e:
        print(e)
        return ""


class Recipe:
    """A recipe object"""
    name: str
    ingredients: [str]
    instructions: str
    image: str
    

if __name__ == '__main__':
    # Example usage
    seasonings = ['salt', 'pepper', 'paprika', 'soy source', 'ketchap']
    items = ['chicken', 'rice', 'broccoli', 'mango', 'italian pasta', 'beef', 'egg']
    print(get_recipes(seasonings, items))

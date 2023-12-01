import os
import openai
from video_to_dynamic.jpg_to_dynamic.clip_interrogator.clip_interrogator_local import clip_interrogator_local

def jpg_to_dynamic(image_path):
    image_description = clip_interrogator_local(image_path)
    
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    messages = [{"role": "user", "content": f"From an AI description of an image, extract the words related to emotions. Only output the words related to emotions. Output it like this:\nThe words related to emotions in the given description are: happy, sad\nHere is the description:\n{image_description}"}]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    response_content = response["choices"][0]["message"]["content"]

    return response_content
    

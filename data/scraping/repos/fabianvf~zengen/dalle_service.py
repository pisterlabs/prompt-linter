# server/dalle_service.py
import openai
import base64
from binascii import a2b_base64


def generate_image(koan):
    try:
        enriched_prompt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system", 
                "content": "You will interpret this koan, and generate a prompt for dalle3 for an image that will prompt reflection in the student targeted by this koan. The image will have no text, it will just be heavily stylized, abstract dreamscape. Your response will contain no content other than the prompt to be passed directly to dalle",
            }, {
                "role": "user", 
                "content": koan
            }]
        )
        enriched_prompt = enriched_prompt_response['choices'][0]['message']['content']
        print(enriched_prompt)

        response = openai.Image.create(
            prompt=enriched_prompt,
            n=1,  # Generate 1 image
            size="512x512",
            response_format="b64_json"
        )

        return enriched_prompt, response['data'][0]['b64_json']


    except Exception as e:
        print(f"Failed to generate image: {e}")
        return None

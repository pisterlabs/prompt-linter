import sys
from dotenv import load_dotenv
import openai
import os
import emoji

load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')

# read the contents of the buffer from standard input
buffer = sys.stdin.read()


def assistant(buffer):
    # program needs to have prevention from a timeout to openAI
    retry_count = 0
    max_retries = 9999

    while retry_count <= max_retries:
        try:
            # call openai api
            response = openai.ChatCompletion.create(
                # model type
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "user", "content": buffer},
                ],
                temperature=1,
                max_tokens=15999,
            )

            response_dict = response.get("choices")
            if response_dict and len(response_dict) > 0:
                prompt_response = response_dict[0]["message"]["content"]

            return prompt_response

        except openai.error.InvalidRequestError as e:
            print(f"API request [InvalidRequestError] failed with error: {e}")
            smiley = emoji.emojize(":smiling_face_with_smiling_eyes:")
            return assistant(prompt=smiley)

        except Exception:
            retry_count += 1


output = assistant(buffer)
print('\n\n## Question\n\n')
print(f"{buffer}")
print('\n\n## Answer\n\n')
print(f"{output}")

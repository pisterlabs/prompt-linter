import openai
import sys
import dotenv
import os

dotenv.load_dotenv()

# Color escape sequences
COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_BLUE = '\033[94m'

openai.api_key = os.getenv('openai')


def complete_text(prompt_user):
    print('[AI]: ')
    if prompt_user:
        prompt_user = f"Act as a professional AI automation system capable of providing concise and accurate answers " \
                      f"to various questions. Focus on understanding the problem or question at hand and provide " \
                      f"brief responses without excessive descriptions. Prompt: {prompt_user}"

        for resp in openai.Completion.create(
                engine='text-davinci-003',  # Use the text-davinci-003 engine for completion
                prompt=prompt_user,
                max_tokens=100,  # Adjust the value as per your requirement
                temperature=0.7,  # Adjust the value as per your preference
                n=1,  # Generate a single response
                stop=None,  # Let the model determine the completion automatically
                stream=True
        ):
            sys.stdout.write(resp.choices[0].text)
            sys.stdout.flush()
    else:
        print(f"{COLOR_RED}[!]{COLOR_RED} please type something.")


while True:
    try:
        prompt = input(f'{COLOR_BLUE}USER:{COLOR_BLUE} [+] ask me something: ')
        complete_text(prompt)
    except openai.APIError as error_api:
        print(f'[!] you miss api key please try to change it to valid api,\nthis link to the openai api key '
              f'https://platform.openai.com/account/api-keys \nthe error message: {error_api}')
        break
    except openai.InvalidRequestError as invalid:
        print(f'{COLOR_YELLOW}[!]{COLOR_YELLOW} something what wrong\nERROR: {invalid}')
    except KeyboardInterrupt:
        print(f'{COLOR_YELLOW}[!]{COLOR_YELLOW} the script close by the user.')
    print('')

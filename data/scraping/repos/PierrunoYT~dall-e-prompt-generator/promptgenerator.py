import sys
import configparser
from nltk.tokenize import word_tokenize
import openai
import traceback
import nltk


# Read the configuration file
config = configparser.ConfigParser()
try:
    config.read('config.ini')
except Exception as e:
    print(f"Error reading the configuration file: {e}")
    sys.exit(1)

# Set your OpenAI API key
openai.api_key = config.get('openai', 'api_key')

generation_counter = 0

def generate_dall_e_prompt(prompt_length):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are operating DALL-E, an image generator. Please provide a description of the image you would like to generate."},
            ]
        )
        print(f"Response: {response}")

        content = response["choices"][0]["message"]["content"]
        trimmed_content = trim_to_length(content, prompt_length)
        return trimmed_content

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None


def generate_prompt_for_keywords(keywords, prompt_length):
    try:
        message_content = "You are operating DALL-E, an image generator. Please provide a description of the image you would like to generate involving '{}'.".format(keywords)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": message_content},
            ]
        )

        print(f"Response: {response}")

        content = response["choices"][0]["message"]["content"]
        trimmed_content = trim_to_length(content, prompt_length, keywords)
        return trimmed_content

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None


def trim_to_length(content, length, keywords=None):
    tokens = word_tokenize(content)
    if len(tokens) <= length:
        return content
    else:
        # Find sentences containing keywords
        keyword_sentences = []
        if keywords:
            for sentence in nltk.sent_tokenize(content):
                if any(keyword in sentence.lower() for keyword in keywords.split()):
                    keyword_sentences.append(sentence)

        # Join selected sentences and trim to length
        trimmed_content = ' '.join(keyword_sentences[:length]) if keyword_sentences else content[:length]
        return trimmed_content


def main():
    print("Menu:")
    print("1. Generate DALL-E prompt")
    print("2. Generate prompt for keyword(s)")

    try:
        choice = input("Enter your choice: ")

        if not choice.isdigit():
            print("Please enter a valid number.")
            sys.exit(1)

        choice = int(choice)

        if choice == 1:
            prompt_length = int(input("Enter the desired prompt length (5-100 words): "))
            if prompt_length < 5 or prompt_length > 100:
                print("Invalid prompt length.")
                sys.exit(1)

            prompt = generate_dall_e_prompt(prompt_length)
        elif choice == 2:
            keywords = input("Enter keyword(s) or sentence: ")
            prompt_length = int(input("Enter the desired prompt length (5-100 words): "))
            if prompt_length < 5 or prompt_length > 100:
                print("Invalid prompt length.")
                sys.exit(1)

            prompt = generate_prompt_for_keywords(keywords, prompt_length)
        else:
            print("Invalid choice.")
            sys.exit(1)

        global generation_counter
        generation_counter += 1

        if prompt is None:
            print("Prompt generation failed.")
        else:
            print(f"Prompt: {prompt}")
            print(f"Total image generations: {generation_counter}")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

#v5

import openai
from getpass import getpass
from openai import ChatCompletion

# Fetch the API key from an environment variable or a config file
# OPENAI_API_KEY = getpass("Please provide your OpenAI API Key:")
#if not OPENAI_API_KEY:
#    raise ValueError("No OpenAI API key found!")

#openai.api_key = OPENAI_API_KEY

VALID_MBTI_TYPES = ["ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
                   "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"]

def get_activities(mbti1, mbti2):
    chat_text = f"Given the MBTI types {mbti1} and {mbti2}, suggest a list of couple's activities that would be ideal for this combination."

    chat_completion = ChatCompletion.create(model="gpt-3.5-turbo", messages=[
    {"role": "user", "content": chat_text}
    ])

    results = chat_completion.choices[0].message.content
    return results

def main():
    print("Welcome to the MBTI Couple's Activity Recommender!")

    mbti1 = input("Enter your MBTI type: ").upper()
    while mbti1 not in VALID_MBTI_TYPES:
        print("Invalid MBTI type. Please enter a valid MBTI type.")
    mbti2 = input("Enter your partner's MBTI type: ").upper()
    while mbti2 not in VALID_MBTI_TYPES:
        print("Invalid MBTI type. Please enter a valid MBTI type.")
        mbti2 = input("Enter your partner's MBTI type: ").upper()

    activities = get_activities(mbti1, mbti2)
    print("\nHere are some suggested couple's activities:")
    print(activities)

if __name__ == "__main__":
    main()
from dotenv import load_dotenv
import os
import argparse
import openai

if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-a", "--animal", help="Animal name", required=True, type=str)
    args = arg_parser.parse_args()
    animal = args.animal

    # Keep some restriction on input
    # As a basic check, we will complain if input if longer that 100 characters
    if len(animal) > 100:
        arg_parser.error("Animal name is too long. Ensure that it has less than 100 characters.")

    # First verify if the given input is a valid animal name
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", 
             "content":"""
             If user input is an animal name, output 'YES'. Else, output 'NO'.
            
             Output Format:
             - YES (if user input is an animal)
             - NO (if user input is not an animal)

             Here are some example inputs:

             USER: dog
             ASSISTANT: YES

             USER: cat
             ASSISTANT: YES

             USER: python
             ASSISTANT: YES

             USER: bottle
             ASSISTANT: NO
             """ },
            {"role": "user", "content": animal},
        ],
        temperature=0,
    )

    animal_check = response['choices'][0]['message']['content']

    # If the output is not valid, give up :(
    if animal_check not in ["YES", "NO"]:
        print("ERROR: Unexpected error from OpenAI API. Recieved output:", animal_check)
        exit(1)
    
    if animal_check == "NO":
        arg_parser.error(f"{animal} is not an animal name.")

    # Suggest 3 petnames
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """
                Suggest 3 pet names for the animal name in user's input.
                """
            },
            {
                "role": "user",
                "content": animal,
            }
        ],
        temperature=0.5,
    )

    print(response['choices'][0]['message']['content'])

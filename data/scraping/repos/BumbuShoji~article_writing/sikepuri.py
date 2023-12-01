import openai
import random
import textwrap

openai.api_key = "sk-9j5bKDjYO2xVVTOoZCH4T3BlbkFJ6fSLT8D2EKMLszop30QKe"

def split_sentence(sentence, max_length=50):
    return textwrap.wrap(sentence, max_length)

def generate_prompt(sentence_parts):
    random.shuffle(sentence_parts)
    return " ".join(sentence_parts)

def ask_question(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

def check_answer(prompt, user_answer):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"{prompt}\n\nUser: {user_answer}\n\nIs the user's answer correct?",
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return "yes" in response.choices[0].text.strip().lower()

def main():
    file_path = input("Enter the path to the .txt file: ")

    try:
        with open(file_path, 'r') as file:
            input_sentence = file.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    sentence_parts = split_sentence(input_sentence)

    while True:
        prompt = generate_prompt(sentence_parts)
        question = ask_question(f"Ask a question about: {prompt}")
        print(f"Question: {question}")

        user_answer = input("Answer: ")
        is_correct = check_answer(prompt, user_answer)

        if is_correct:
            print("Your answer is correct!")
        else:
            print("Your answer is incorrect, please try again.")

if __name__ == "__main__":
    main()

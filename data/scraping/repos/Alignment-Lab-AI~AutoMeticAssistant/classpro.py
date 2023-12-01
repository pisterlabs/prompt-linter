import csv
import os
import openai

openai.api_key = 'KEYHERE'

def generate_response(system_prompt, question, answer, model="gpt-4"):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"QUESTION: {question}\nANSWER: {answer}"},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=2500,
        temperature=0.7,
    )

    return response.choices[0].message['content']

def main():
    while True:
        with open('questions.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip the header
            question_pairs = list(reader)

        new_pairs = []
        for i, pair in enumerate(question_pairs):
            question, answer = pair
            system_prompt = "The following is an example of a language model opting to avoid answering a question without outright refusing. Please reply with another example-pair of this behavior ONLY and in the EXACT same tab-separated format."
            new_pair = generate_response(system_prompt, question, answer)
            new_question, new_answer = new_pair.split('ANSWER: ')
            new_question = new_question.lstrip('QUESTION: ')  # Remove the 'QUESTION: ' prefix
            new_pairs.append((new_question, new_answer.strip()))  # Add the new pair to the list
            print(f"Question: {question}\nAnswer: {answer}\nNew Pair: {new_pair}\n")
            print(f"Total rows in CSV (including header): {len(question_pairs) + 1 + i}\n")

            # Save the new rows to CSV every 10 iterations
            if (i+1) % 10 == 0:
                with open('questions.csv', 'a') as file:
                    writer = csv.writer(file)
                    writer.writerows(new_pairs)
                    new_pairs = []  # Reset the new_pairs list after writing to file

        # Write any remaining pairs to file
        if new_pairs:
            with open('questions.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerows(new_pairs)
            new_pairs = []  # Reset the new_pairs list after writing to file

if __name__ == "__main__":
    main()

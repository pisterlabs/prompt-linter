import json
import os
import openai
import re
import threading
import random

def custom_tokenize(sentence):
    return re.findall(r'\b\d+-\d+\b|\b\w+\b|[\'?]', sentence)

def extract_strings_from_nested_list(lst):
    result = []
    for x in lst:
        if isinstance(x, str):
            result.append(x)
        elif isinstance(x, list):
            result.extend(extract_strings_from_nested_list(x))
    return result

def timed_chat_completion(original_question, result_container, where_key, num_questions):
    try:
        prompt = (
            f"Can you provide {num_questions} different version(s) of the question "
            f"'{original_question}' without any introductory text, "
            f"quotation marks or additional comments? "
            f"Just the questions, please. "
            f"Make sure {where_key} does not change and that "
            f"a natural language to SQL query AI model understands."
        )

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        altered_question_content = completion.choices[0].message["content"]
        result_container.append(altered_question_content)
    except Exception as e:
        print(f"An error occurred: {e}")

# Updated to accept num_questions as an argument
def generate_altered_questions(original_question, original_where_key, num_questions, timeout=20):
    while True:  # Keep retrying until successful
        result_container = []
        thread = threading.Thread(target=timed_chat_completion, args=(original_question, result_container, original_where_key, num_questions))
        thread.daemon = True  # Set thread as a daemon
        thread.start()
        thread.join(timeout=timeout)

        if result_container:
            altered_question_content = result_container[0]
            return [q.split('. ')[1] if '. ' in q else q for q in altered_question_content.split('\n')]
        else:
            print(f"Timed out after {timeout} seconds. Retrying...")

def update_record(record, new_question):
    new_record = record.copy()
    question_tokens = custom_tokenize(new_question)
    
    def check_for_space(token1, token2):
        original_index_end = new_question.find(token1) + len(token1)
        original_index_start = new_question.find(token2)
        return original_index_start - original_index_end
    
    new_record.update({
        'question': new_question,
        'question_tok': question_tokens,
        'question_tok_space': [
            ' ' if check_for_space(question_tokens[i], question_tokens[i+1]) == 1 else '' 
            for i in range(len(question_tokens) - 1)
        ] + ['']
    })
    
    return new_record

if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    original_records = []
    with open('data/test_tok.jsonl', 'r') as file:
        for line in file:
            original_records.append(json.loads(line.strip()))
            
    total_questions = 1000
    random.shuffle(original_records)
    original_records = original_records[:total_questions]
    
    for num_questions in range(4, 6):
        altered_records = []
        for index, original_record in enumerate(original_records):
            where_key = extract_strings_from_nested_list(original_record['sql']['conds'])
            altered_questions = generate_altered_questions(original_record['question'], where_key, num_questions)
            new_altered_records = [update_record(original_record, q.replace('\"', '')) for q in altered_questions]

            # Only keep records where the "question" is not empty and starts with a letter of the alphabet
            new_altered_records = [record for record in new_altered_records if record['question'] and record['question'][0].isalpha()]

            altered_records.extend(new_altered_records)

            print(f"Round {num_questions}: Completed question {index + 1} of {total_questions}. Remaining: {total_questions - (index + 1)}")

        # Save to different files
        with open(f'data/altered_questions_{num_questions}.jsonl', 'w') as new_file:
            new_file.writelines(json.dumps(record) + '\n' for record in altered_records)
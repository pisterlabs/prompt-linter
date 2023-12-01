import os
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]
answers_path = '.././data/artificial/'

def read_txt_file(file_path):
    """
    Reads the content of a text file and returns it as a string.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the text file as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    return file_content

def get_completion(prompt, model="gpt-3.5-turbo"):  # "gpt-3.5-turbo", "gpt-4"
    """
    Generates a completion response for a given prompt using the OpenAI ChatGPT model.

    Args:
        prompt (str): The prompt text to generate a completion for.
        model (str, optional): The model to use for completion generation. Defaults to "gpt-3.5-turbo".

    Returns:
        str: The generated completion response.
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def save_string_to_txt_file(answer_path, string):
    """
    Saves the given string to a text file.

    Args:
        answer_path (str): The path to the output text file.
        string (str): The string to be saved.

    Returns:
        None
    """
    with open(answer_path, 'w', encoding='utf-8') as file:
        file.write(string)

def process_essay_answer_language_classification(folder_path):
    """
    Process files in the specified folder and generate language classification results for essay answers.

    Args:
        folder_path (str): The path to the folder containing the files.

    Returns:
        None
    """
    tests_results = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('essay_answer.txt'):
            answer_path = os.path.join(folder_path, file_name)
            
            essay_answer = read_txt_file(answer_path)
            prompt = "This is an essay answer. Say 'Portuguese' if the answer is in Portuguese, or 'Non-Portuguese' if the answer is in any other language." + "\n\n---\n\n" + essay_answer 
            response = get_completion(prompt)
            tests_results.append(file_name + ": " + response)
            print(file_name + ": " + "Done!")
    
    # Save tests results to a text file
    test_results_path = os.path.join('./', 'essay_language_classification_tests_results.log')
    save_string_to_txt_file(test_results_path, '\n'.join(tests_results))
    print('Tests run successful. Check report at', test_results_path)
    
process_essay_answer_language_classification(answers_path)
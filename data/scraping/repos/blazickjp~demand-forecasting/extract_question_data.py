import re
import json
import openai
import uuid
import logging
from tqdm import tqdm
from requests.exceptions import HTTPError
import time

# Constants
MODEL_NAME = "gpt-4"
TEMPERATURE = 0.0
MAX_QUESTION_NUMBER = 50

# Logging setup
logging.basicConfig(filename="question_parser.log", level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s")


def parse_practice_problems_to_json(content):
    """
    Given a content string, this function constructs a prompt to send to the OpenAI API, 
    sends the prompt, and attempts to parse the response into a Python dictionary.

    Args:
        content (str): The content string to be parsed into JSON.

    Returns:
        dict: A dictionary representing the parsed content.
    """

    guid = str(uuid.uuid4())
    delay = 0.1  # Start with a small delay
    max_delay = 60
    prompt = f"""
Parse the following unstructured questions into json and return the json object with the following structure:
{{
    "id": {guid},
    "question": "QUESTION EXTRACTED FROM CONTENT",
    "options": {{
        "A": "OPTION A",
        "B": "OPTION B",
        "C": "OPTION C"
    }},
    "data": {{}}
}}
For questions with associated data, the 'data' field should include a description (if provided) and a table (if provided). 
The table should be represented as an object with 'headers' and 'rows' fields, and each row should be an object with keys 
matching the headers and values representing the cell contents. If a cell is empty or not applicable, it should be represented 
with an empty string. Keys should never be empty strings.

EXAMPLE 1:
CONTENT: 1. Published ratings on stocks ranging from 1 (strong sell) to 5 (strong buy) are 
examples of which measurement scale?
A. Ordinal
B. Continuous
C. Nominal
JSON:
{{
    "id": "d64adf0e-f821-4f9f-af97-e28d98d0317a",
    "question": "Published ratings on stocks ranging from 1 (strong sell) to 5 (strong buy) are examples of which measurement scale?",
    "options": {{
        "A": "Ordinal", 
        "B": "Continuous", 
        "C": "Nominal"
        }},
    "data": ""
}}

EXAMPLE 2:
CONTENT:"7. Each individual row of data in the table can be best characterized as:\nA. panel data.\nB. time-series data.\nC. cross-sectional data.\nDATA: An equity analyst gathers total returns for three country equity indexes over the \npast four years. The data are presented below.\n\nLearning Module 2 \nOrganizing, Visualizing, and Describing Data\n152\nTime Period\nIndex A\nIndex B\nIndex C\nYear t–3\n15.56%\n11.84%\n-4.34%\nYear t–2\n-4.12%\n-6.96%\n9.32%\nYear t–1\n11.19%\n10.29%\n-12.72%\nYear t\n8.98%\n6.32%\n21.44%'
JSON:
{{
  "id": "d29ae7d0-b227-410f-b262-02009051a477",
  "question": "Each individual row of data in the table can be best characterized as:",
  "options": {{
    "A": "panel data",
    "B": "time-series data",
    "C": "cross-sectional data"
  }},
  "data": {{
    "description": "An equity analyst gathers total returns for three country equity indexes over the past four years.",
    "table": {{
      "headers": ["Time Period", "Index A", "Index B", "Index C"],
      "rows": [
        {{ "Time Period": "Year t-3", "Index A": "15.56%", "Index B": "11.84%", "Index C": "-4.34%" }},
        {{ "Time Period": "Year t-2", "Index A": "-4.12%", "Index B": "-6.96%", "Index C": "9.32%" }},
        {{ "Time Period": "Year t-1", "Index A": "11.19%", "Index B": "10.29%", "Index C": "-12.72%" }},
        {{ "Time Period": "Year t", "Index A": "8.98%", "Index B": "6.32%", "Index C": "21.44%" }}
      ]
    }}
  }}
}}

EXAMPLE 3:
CONTENT: '23. The annual returns for three portfolios are shown in the following exhibit. Portfo-\nlios P and R were created in Year 1, Portfolio Q in Year 2.\n\xa0\nAnnual Portfolio Returns (%)\n\xa0\nYear 1\nYear 2\nYear 3\nYear 4\nYear 5\nPortfolio P\n-3.0\n4.0\n5.0\n3.0\n7.0\nPortfolio Q\n-3.0\n6.0\n4.0\n8.0\nPortfolio R\n1.0\n-1.0\n4.0\n4.0\n3.0\nThe median annual return from portfolio creation to Year 5 for:\nA. Portfolio P is 4.5%.\nB. Portfolio Q is 4.0%.\n\nPractice Problems\n157\nC. Portfolio R is higher than its arithmetic mean annual return.'
JSON:
{{
    "id": "23",
    "question": "The median annual return from portfolio creation to Year 5 for:",
    "options":{{
        "A": "Portfolio P is 4.5%.",
        "B": "Portfolio Q is 4.0%.",
        "C": "Portfolio R is higher than its arithmetic mean annual return."
    }},
    "data": {{
        "table": {{
            "headers": ["Portfolio", "Year 1","Year 2","Year 3","Year 4","Year 5"],
            "rows": [
                {{"Portfolio": "P", "Year 1": "-3.0", "Year 2": "4.0", "Year 3": "5.0", "Year 4": "3.0", "Year 5": "7.0"}},
                {{"Portfolio": "Q", "Year 1": "", "Year 2": "3.0", "Year 3": "6.0", "Year 4": "4.0", "Year 5": "8.0"}},
                {{"Portfolio": "R", "Year 1": "1.0", "Year 2": "-1.0", "Year 3": "4.0", "Year 4": "4.0", "Year 5": "3.0"}}
            ]
        }}
    }}
}}

CONTENT: {content}
JSON:"""

    while delay <= max_delay:
        try:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful AI that parses unstructured text into JSON"},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE
            )
            parsed_json = json.loads(
                response['choices'][0]['message']['content'])
            return parsed_json

        except json.JSONDecodeError:
            print(f"Failed to parse with GP3.5 Turbo. Trying with GPT-4.")
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful AI Assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=TEMPERATURE
                )
                parsed_json = json.loads(
                    response['choices'][0]['message']['content'])
                print("Successfully parsed with GPT-4!")
                return parsed_json
            except json.JSONDecodeError:
                print(
                    f"Failed to parse the following content into JSON:\n{content}")
                return None
        except:
            print("Rate limit error detected, waiting before retrying...")
            print(f"Rate limit exceeded, sleeping for {delay} seconds...")
            time.sleep(delay)
            delay *= 2
            continue

    raise Exception("Max delay exceeded")


def find_practice_problems(data):
    """
    Searches the JSON data for the "Practice Problems" section and returns a list of the practice problems,
    each with the "learning_objectives" and "learning_module_name" field from the parent node.

    Args:
        data (dict): The JSON data to be searched.

    Returns:
        list: A list of practice problems found in the data, each with the "learning_objectives" and "learning_module_name" field from the parent node.
    """

    practice_problems = []

    # Recursive function to search for practice problems in the JSON data
    def search_practice_problems(data, learning_objectives=None, learning_module_name=None):
        if isinstance(data, dict):
            if 'title' in data and data['title'] == "Practice Problems":
                # Add the "learning_objectives" and "learning_module_name" field to the practice problem
                data["learning_objectives"] = learning_objectives
                data["learning_module_name"] = learning_module_name
                practice_problems.append(data)
            else:
                new_learning_objectives = data.get(
                    "learning_objectives", learning_objectives)
                # Check if the title starts with "Learning Module" and if so, update the learning module name
                if data.get('title', '').startswith("Learning Module"):
                    learning_module_name = data['title']
                for key in data:
                    search_practice_problems(
                        data[key], new_learning_objectives, learning_module_name)
        elif isinstance(data, list):
            for item in data:
                search_practice_problems(
                    item, learning_objectives, learning_module_name)

    search_practice_problems(data)
    return practice_problems


def extract_question(text, question_number):
    """
    Extracts a question with its corresponding number from the given text.

    Args:
        text (str): The text to extract the question from.
        question_number (int): The number of the question to extract.

    Returns:
        str: The extracted question.
    """
    question_match = re.search(
        rf"\n{question_number}\.\s.*?A\..*?B\..*?C\.", text, re.DOTALL)
    if question_match:
        question = question_match.group().strip()
        return question
    else:
        return None


def parse_questions(text):
    """
    Parses the content from the "Practice Problems" text of the CFAI pdf file and 
    returns a list of the end of chapter questions.
    """
    questions = []
    info_text = ""
    info_first_q = 0
    info_second_q = 0
    max_question_number = 50  # You can change this to the highest expected question number
    # Remove page headers
    text = re.sub(r"Practice Problems \d+", "", text)
    question_end = None
    # text = re.sub(rf'Learning Module {mod_number}\s+{mod_name}', "", text)

    for question_number in range(1, max_question_number + 1):
        # Look for the info text related to the current question
        info_match = re.search(
            rf"The following information relates to questions[\s\n]*{question_number}-\d+", text)
        # If info text is found, store it for future use
        if info_match:
            info_range = re.findall(r'\d+', info_match.group())
            info_first_q = int(info_range[0])
            info_second_q = int(info_range[1])
            info_start = info_match.end()

            if info_first_q <= question_number <= info_second_q:
                # Find the next question
                next_question_match = re.search(
                    rf"\n{question_number}\.\s.*?A\..*?B\..*?C\.", text, re.DOTALL)

                if next_question_match:
                    info_end = next_question_match.start()
                else:
                    info_end = len(text)

                # Extract the info text
                info_text = text[info_start:info_end].strip()

        # Search for the question with answers (A., B., and C.)

        text = text[question_end:] if question_end else text
        question_match = re.search(
            rf"\n{question_number}\.\s.*?A\..*?B\..*?C\.", text, re.DOTALL)

        if question_match:
            question_start = question_match.start()

            # Find the next question
            next_question_number = question_number + 1
            next_question_match = re.search(
                rf"\n{next_question_number}\.\s.*?\nA\..*?\nB\..*?\nC\.", text, re.DOTALL)

            if next_question_match:
                if question_match.end() > next_question_match.start():
                    text = text[next_question_match.start():]
                    continue
                else:
                    question_end = next_question_match.start()
            else:
                question_end = len(text)

            # Extract the question
            question = text[question_start:question_end].strip()
            matches = [match for match in re.finditer(r"C\.\s.*?\.", question)]
            last_match = matches[-1] if matches else None
            question = question[:last_match.end()] if last_match else question

            # Prepend the info text, if any, to the question
            if info_text:
                if info_first_q <= question_number <= info_second_q:
                    question = f"{question}\nDATA: {info_text}"
                # else:
                #     info_text = ""
                #     text = text[info_end:]

            questions.append(question)

    return questions


def main(file, output_file):

    # Load the JSON object
    with open(file, 'r') as f:
        data = json.load(f)

    # Find the practice problems
    practice_problems = find_practice_problems(data)

    # Parse the practice problems into questions
    questions = []
    for problem in practice_problems:
        content = problem['content']
        learning_objectives = problem.get('learning_objectives', [])
        learning_module_name = problem.get('learning_module_name', '')
        parsed_questions = parse_questions(content)
        # Add the "learning_objectives" and "learning_module_name" to each question
        for question in parsed_questions:
            question_dict = {
                'content': question,
                'learning_objectives': learning_objectives,
                'learning_module_name': learning_module_name,
            }
            questions.append(question_dict)

    # Parse the questions into JSON
    questions_data = []
    for question in tqdm(questions, desc="Parsing questions"):
        # print(question)
        question_data = parse_practice_problems_to_json(question['content'])
        # print(question_data)
        # break
        if question_data:
            question.update({"parsed_question": question_data})
            questions_data.append(question)

    # Save the questions data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(questions_data, f, indent=4)


if __name__ == "__main__":
    file = "./output/level_1_volume_1.json"
    output_file = "questions_data.json"
    main(file, output_file)

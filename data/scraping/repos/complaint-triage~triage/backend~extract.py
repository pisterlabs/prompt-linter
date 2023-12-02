import re
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")
import json
from json.decoder import JSONDecodeError

from langchain.chat_models import AzureChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    HumanMessage
)



# Functions

def case_insensitive_search(text, pattern):
    """
    Perform a case-insensitive search for the given pattern in the text.

    Args:
    text (str): The text to search within.
    pattern (str): The pattern to search for.

    Returns:
    bool: True if a match is found, False otherwise.
    """
    # Use re.IGNORECASE flag to make the search case-insensitive
    if re.search(pattern, text, re.IGNORECASE):
        return True
    else:
        return False

# offset_category can either be "start" or "end"
def find_word_substrings(string, substring, offset_category):
    pattern = r'\b{}\b'.format(re.escape(substring))
    match = re.search(pattern, string)
    
    if match:
        if offset_category == "start":
            return match.start()
        else:
            return match.end()
    else:
        return -1  # Return -1 if the word is not found

def extract_sentence_by_char_offset(text, char_offset):
    # Define a pattern to match sentence boundaries (e.g., periods, exclamation marks, question marks)
    sentence_pattern = r'[^.!?]*[.!?]'
    sentences = re.findall(sentence_pattern, text)

    total_chars = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        if total_chars + sentence_length > char_offset:
            return sentence.strip()  # Remove leading/trailing spaces
        total_chars += sentence_length

    # If the character offset is beyond the end of the text, return the last sentence
    return sentences[-1].strip()

# Accepts the filepath of a legal guidelines document and returns json of its the location, scope and subject of compliant. 
def extract_criteria(file_path, model):

    criteria_dict = {}

    try:
        with open(file_path, 'r') as complaint_guide:
            # Read the contents of the file
            text = complaint_guide.read()
            guide_name = os.path.basename(file_path)


            prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
You are a lawyer that answers questions about the criteria of valid complaints of a given Australian legal guideline.   
You must only output answers with the supplied allowed values and format for each question.
                                                      
                                                                                                            
            """),

            HumanMessagePromptTemplate.from_template("""
Read the complaint and answer the questions about criteria:

                <<< Legal Guideline >>> 
                ####
                {legal_guideline}
                ####
  
                <<< Questions >>>  
                ####
1. Produce an array of locations this guideline applies in. Allowed values: "NSW", "ACT", "QLD", "SA", "TAS", "VIC", "WA" and "Overseas".
2. State the time limitation for this guideline or “None”.
3. Produce an array of the valid subjects of the complaint for this guideline.
                ####
            """)
            ])
            
            answers = model(prompt.format_prompt(
                legal_guideline=text
            ).to_messages()).content


            # convert answer string format to json object
            # Split the string into individual dot points
            dot_points = re.split(r'\d+\.', answers)

            # Remove extra spacing and newline characters
            dot_points = [point.strip() for point in dot_points if point.strip()]

            # Print the extracted dot points
            try:
                location_arr = json.loads(dot_points[0])
                time_limitations = dot_points[1]
                subjects_arr = json.loads(dot_points[2])

                criteria_dict["locations"] = location_arr
                criteria_dict["time_limitations"] = time_limitations
                criteria_dict["subjects"] = subjects_arr

                return criteria_dict

            except JSONDecodeError:
                return None

    except FileNotFoundError:
        print(f"The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



# Global Variables

# States:
# ACT
# queens
# nsw
# tasmania
# victoria
# western aus
# south aus
states = ["NSW", "ACT", "QLD", "SA", "TAS", "VIC", "WA"]



# Execution

load_dotenv('setting.env')
load_dotenv('secret.env')

model = AzureChatOpenAI(temperature=0, deployment_name="gpt-4-32k")

# Specify the parent directory you want to search in
parent_directory = "./pathways"
#filepath = "./pathways/Fair Work Commission.txt"

database_directory = "./json_db"

print("\n\n\n")

# Loop through all child directories, convert all legal guidelines to json criteria
for complaint_guide in os.listdir(parent_directory):
    if ".txt" in complaint_guide:

        guide_text_name = complaint_guide[:-4]
        print(guide_text_name)
        print("\n")

        guide_path_name = os.path.join(parent_directory, complaint_guide)
        
        criteria = extract_criteria(guide_path_name, model)
        if criteria is None:
            raise ValueError("Failed to convert ", complaint_guide)

        # Save criteria as json object
        json_path = os.path.join(database_directory, guide_text_name + ".json")

        with open(json_path, 'w') as json_file:
            json.dump(criteria, json_file, indent=4)


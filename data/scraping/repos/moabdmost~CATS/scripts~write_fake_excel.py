""""Writes fake essays to an Excel file using openai api and openpyxl pachages.

This script is used to generate a number of fake essays based on a given prompt. 
Then, it will save them to an Excel file.
It is worth noting that this script is designed to avoid some resource-related errors that were encountered.
Those errors include internet connection errors, interruption errors, and so on.

If it didn't generate the number of essays you want. Just run the script again and don't change any parameters.
It will print "Mission Accomplished!" when the number of essays is generated.
Excel/CSV data files to be saved in an archive folder for record. 
See excel_to_format.py for the rest of the preprocessing pipeline.
Consider a naming format to keep track of the archive files.

Author: Mohamed Mostafa
"""


import openpyxl
import openai
import os
import time

prompt="""
Type of essay:	Persuasive/ Narrative/Expository
Grade level:	8
Average length of essays:	350 words
Prompt: More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. 
Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.
  """
path = '/workspace/' # Main folder
file_name = 'archive/FILE_NAME.xlsx'
# You can get this key from https://platform.openai.com/account/api-keys
secret_key = "YOUR_SECRET_OPENAI_API_KEY"
number = 20


def write_fake_to_excel(excel_file_path: str, open_api_key: str, prompt: str, generate_number: int, beggining_from:int =2):
    """
    Writes fake essays to an Excel file.
    
    Parameters:
    - excel_file_path: str, the path to the Excel file
    - open_api_key: str, the API key for OpenAI
    - prompt: str, the prompt for generating the essays
    - generate_number: int, the number of essays to generate
    - beggining_from: int, the starting index for the essays (default is 2)
    """
    openai.api_key = open_api_key
    # required number is intended to be a fixed number.
    required_number = beggining_from + generate_number - 2 
    print("required_number is " + str(required_number))
    
    if not os.path.exists(excel_file_path):
        print("file does not exist, starting the generation")
        workbook = openpyxl.Workbook()
        worksheet = workbook.active

        worksheet['A1'] = 'essay_id'
        worksheet['B1'] = 'essay'
        worksheet['C1'] = 'label'
        
        essay_generator(excel_file_path, prompt, worksheet, workbook, required_number, beggining_from)
    else:
        print("file exists, checking if it's complete")
        workbook = openpyxl.load_workbook(excel_file_path)
        worksheet = workbook.active
        max_row_index = 1
        
        for row in range(1, required_number + 2):
            if worksheet['A' + str(row)].value is not None:
                max_row_index = max(max_row_index, row)
        print("Last essay is " + str(max_row_index - 1))
        
        if max_row_index == required_number + 1:
            print("Mission Accomplished!")
            workbook.save(excel_file_path)
        else:
            print("file is not complete, completing the generation")
            essay_generator(excel_file_path, prompt, worksheet, workbook, required_number, max_row_index+1)


def essay_generator(excel_file_path: str, prompt: str, worksheet, workbook, required_number: int, beggining_from: int):
    """
    Generates essays based on a given prompt and writes them to an Excel file.

    Args:
        excel_file_path (str): The path to the Excel file.
        prompt (str): The prompt for generating the essays.
        worksheet: The worksheet object for writing data.
        workbook: The workbook object for saving the Excel file.
        required_number (int): The number of essays to generate.
        beggining_from (int): The starting index for writing the essays.

    Returns:
        None
    """

    print("Generating essays...")

    try:
        for i in range(beggining_from, required_number + 2):
            print("Working on " + str(i - 1))
            # Comment response to test the architecture and the paths.
            response = openai.ChatCompletion.create(
                # replace with the model you want to use
                model="gpt-3.5-turbo",  
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                # adjust the randomness of the completion
                temperature=1, 
            )

            worksheet['A' + str(i)] = i-1
            worksheet['B' + str(i)] = response.choices[0].message.content # "done" for testing
            worksheet['C' + str(i)] = 0

            print("Essay " + str(i - 1) + " is Done!")
            workbook.save(excel_file_path) # save the file every time to avoid any data loss in case of interruptions or errors.
            # time.sleep(1)   #Add this delay to test the architecture.

    # handle any sudden connection errors or interruptions that might occur during the process.
    except (ConnectionError, InterruptedError, Exception) as e:
        print("An exception occurred:", str(e))

    # in case of any interruption, pick up where we left off.
    finally:
        # Close the Excel file
        print("Done this phase!")
        # workbook.save(excel_file_path)
        write_fake_to_excel(excel_file_path, secret_key, prompt, required_number)

# example execution, uncomment, for testing
write_fake_to_excel(path + file_name, secret_key, prompt, number)

import web_scrape  
import os.path
import config
import doc_summarization 
from datetime import date
import json
import time
import openai
import traceback
def error_logger(e, url):
    with open("error_log_summarization.txt", "a") as file:
        file.write(f"Invalid request error retrieving: {url}")
        file.write("Here is the error:\n")
        file.write(traceback.format_exc())
        file.write("\n")

def write_json_to_file(json_data, filename, directory):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Construct the full file path
    filepath = os.path.join(directory, filename)

    # Write the JSON data to a .json file
    with open(filepath, 'w') as file:
        json.dump(json_data, file, indent=4)

with open ('websites.txt', 'rt') as myfile:  # Open websites.txt for reading
    for myline in myfile:              # For each line, read to a string,
        start_time = time.time()

        url = myline.strip()        # Each line is a new URL to open and scrape
        json_data = {} # Initialize the JSON file. 
        json_data["URL"] = url
        # Step 1: Webpage Scraping
        page_text = web_scrape.extract_text_from_url(url)
        
        # Step 2: Obtain a unique ID for the webpage URL
        webpage_guid = doc_summarization.encode_to_guid(url)
        print(f"webpage GUID: {webpage_guid} and website URL: {url}")
        out_file_name = webpage_guid + ".txt"

        if page_text:
            # Save the extracted website text to extracted_websites. 
            save_path = 'extracted_websites/'
            completeName = os.path.join(save_path, out_file_name)         
            file1 = open(completeName, "w")
            file1.write(page_text)
            file1.close()

            # Create a summary for each webpage and write to JSON
            try:
                directory = "files_to_index/"
                json_data["Time stamp"] = str(date.today())
                json_data["Summary"] = doc_summarization.summarize_doc(document=page_text)
                json_filename = webpage_guid + ".json"
                write_json_to_file(json_data=json_data, filename=json_filename, directory=directory)
            except openai.error.InvalidRequestError as e:
                error_logger(e, url=url)
            except Exception as e:
                error_logger(e, url=url)
            
            end_time = time.time()

            execution_time = end_time - start_time
            print("The above website took {} seconds to create.".format(execution_time))
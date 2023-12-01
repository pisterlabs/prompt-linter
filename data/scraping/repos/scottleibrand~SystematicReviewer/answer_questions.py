# Description: This script answers questions using the embeddings of the context and the question.
import os
import re
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import json
import sys
import numpy as np
import tiktoken
import openai
import time

def search_embeddings(df, question_string, n):
    question_embedding = get_embedding(
        question_string,
        engine="text-embedding-ada-002"
    )
    # Add a similarity column to the dataframe calculated using the cosine similarity function
    df["similarities"] = df.embedding.apply(lambda x: cosine_similarity(x, question_embedding))
    
    res = (
        df.sort_values("similarities", ascending=False)
        .head(n)
        .text
        # Get all the elements of the list
        .apply(lambda x: x)
    )
    return res

def convert_to_json(res):
  # Convert the dataframe to a list of dictionaries, where each dictionary
  # represents a message with 'text' as the key and the message text as the value
  ticket = [{'text': text} for text in res]

  # Create a JSON object with the list of dictionaries as the value for the 'messages' key
  result = {'ticket': ticket}

  # Return the JSON object
  return result



def get_nth_result(results, n):
  # Get the list of messages from the results
  ticket = results['ticket']

  # Get the Nth message from the list
  nth_message = ticket[n]

  # Return the text of the Nth message
  return nth_message['text']

def ask_gpt(prompt, model_engine="text-davinci-003", max_tokens=100, temperature=0):
    # Get the API key from the environment variable
    api_key = os.environ["OPENAI_API_KEY"]
    openai.api_key = api_key

    # Set the model to use, if not specified
    if model_engine is None:
        model_engine = "text-davinci-003"

    # Generate completions
    for retry in range(3):
        try:
            completions = openai.Completion.create(
                engine=model_engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            break
        except openai.error.ServiceUnavailableError as e:
            if retry < 2:
                time.sleep(retry * 5)
            else:
                answer = "ServiceUnavailableError"
                return answer
    
    # Get the summary from the first completion
    answer = completions.choices[0].text

    return answer

def generate_prompt(question_string, context, question_type, categories):
    # Create a prompt for the question
    prompt = "Given the following context:\n" + context + \
        "\nAnswer the following question:\n" + question_string + "\n"
    # If the question type is "Categorization", add the categories as the choices
    if question_type == "Categorization":
        prompt = prompt + "Please answer: '"
        for category in categories:
            prompt = prompt + str(category) + "', '"
        prompt = prompt + "None of the above'.\n"
    else:
        "If the context is not relevant to the question, reply with 'Not relevant'.\n"
    return prompt
        

def main(papers_csv_file, json_file, question_string_or_file, n, min_sections):
    # Process the papers csv file
    papers_df = pd.read_csv(papers_csv_file)
    # Get the absolute path of the papers csv file
    papers_csv_file = os.path.abspath(papers_csv_file)
    # Get the full path of the papers csv file
    path = os.path.dirname(papers_csv_file)
    print(path)
    # Get the base file name of the papers csv file
    base_file_name = os.path.basename(papers_csv_file)
    # Remove the extension
    base_file_name = os.path.splitext(base_file_name)[0]
    full_file_name = f"{path}/{base_file_name}.top_{n}_answers_{min_sections}_sections.csv"
    
    #TODO: If the file already exists, read it so we can skip the questions that are already answered for a given paper

    urls = papers_df['ArticleURL']
    for url in urls:
        # Skip "nan" URLs
        if pd.isna(url):
            #answers.append('')
            continue
        print(f"Processing {url}...")
        base_file_name = os.path.basename(url)
        if base_file_name == '':
            base_file_name = url.rsplit('/', 2)[-2]
        # If it's still longer than 100 characters, truncate it
        if len(base_file_name) > 100:
            base_file_name = base_file_name[:100]

        # Remove non-alphanumeric, non-period and non-underscore characters from the file name
        base_file_name = re.sub(r'[^\w_.]', '', base_file_name)
        
        # Load the json file
        with open(json_file) as f:
            data = json.load(f)

        # If the question string is a file, read it as a csv
        if os.path.isfile(question_string_or_file):
            # Read the file as a csv
            question_df = pd.read_csv(question_string_or_file)
        else:
            # Create a dataframe with a single row containing the question string
            question_df = pd.DataFrame({'Question': [question_string_or_file]})
            # Add a dummy column for the question type
            question_df['Type'] = 'Open-ended'
                    
        # If the csv has a column named "SectionsFound", skip if less than min_sections
        if 'SectionsFound' in papers_df.columns:
            # Look up the row with the same URL as the current URL
            section_df = papers_df[papers_df['ArticleURL'] == url]
            # Extract the value from the SectionFound column as a string
            if len(section_df) != 0:
                section_found = section_df['SectionsFound'].values[0]
                if section_found < min_sections:
                    #answers.append('')
                    print(f"Skipping {url} because it has only {section_found} sections.")
                    continue

        # Loop through the rows of the questions csv
        for index, row in question_df.iterrows():
            # Get the question string from the row
            question_string = row['Question']
            # If the question string is empty, skip this row
            if pd.isna(question_string):
                continue
            # Add a column header to the dataframe for the question string, if not already present
            if question_string not in papers_df.columns:
                papers_df[question_string] = ''
            # Get the question type from the row
            if 'Type' in question_df.columns:
                question_type = row['Type']
            else:
                question_type = 'Open-ended'
            categories = []
            # If the question type is "Categorization", read in all of the columns starting with "Category" into a list
            if question_type == "Categorization":
                for col in question_df.columns:
                    if col.startswith("Category"):
                        if not pd.isna(row[col]):
                            categories.append(row[col])

            # Store the intermediate results in a list
            intermediate_results = []

            # TODO: Check if the answer is already in the csv file for this question and paper

            # Use this row's value from the Title and Abstract column of the csv, if present, as the context
            if 'Title' in papers_df.columns:
                # Look up the row with the same URL as the current URL
                title_df = papers_df[papers_df['ArticleURL'] == url]
                # Extract the value from the Title column as a string
                if len(title_df) != 0:
                    context = title_df['Title'].values[0]

            if 'Abstract' in papers_df.columns:
                # Look up the row with the same URL as the current URL
                abstract_df = papers_df[papers_df['ArticleURL'] == url]
                # Extract the value from the Abstract column as a string
                if len(abstract_df) != 0:                    
                    context = context + "\n" + abstract_df['Abstract'].values[0]

                    #print(context)
                    prompt = generate_prompt(question_string, context, question_type, categories)
                    #print(prompt)
                    answer = ask_gpt(prompt)
                    #print(answer)
                    intermediate_results.append(answer)

            # Find all records whose "file_name" key contains the base_file_name extracted from the URL
            matching_records = []
            print(f"Searching for records with file names containing {base_file_name}...")
            for record in data:
                if base_file_name in record['file_name']:
                    #print(f"Found record with file name {record['file_name']}.")
                    matching_records.append(record)
                
            print(f"Found {len(matching_records)} matching records.")
            if len(matching_records) == 0:
                print("No matching records found. Skipping...")
                #answers.append('')
                continue
            # Load all of those matching records into a pandas dataframe
            matching_df = pd.DataFrame(matching_records)

            # Search the embeddings
            matching_res = search_embeddings(matching_df, question_string, n)
            print(matching_res)



            # Loop through each result
            for i in range(len(matching_res)):
                # Convert the result to a JSON object
                results = convert_to_json(matching_res)
                result = get_nth_result(results, i)
                results_string = json.dumps(result)
                #print(results_string)

                # Get the token length of the string
                enc = tiktoken.get_encoding("gpt2")
                tokens = enc.encode(results_string)
                token_count = len(tokens)
                # print the length of the string in characters and tokens
                print(f"String length: {len(results_string)} characters, Token count: {token_count}")
                #print(results_string)

                context = results_string

                prompt = generate_prompt(question_string, context, question_type, categories)
                #print(prompt)
                answer = ask_gpt(prompt)
                print(answer)
                intermediate_results.append(answer)

            # Ask GPT to provide an overall answer based on the intermediate results
            prompt = "Given the following " + str(len(intermediate_results)) + " answers:\n" + \
                "\n".join(intermediate_results) + \
                "\nPlease provide an overall answer to the question, if possible:\n" + \
                question_string + "\n" + \
                "\nIf none of the responses above answer the question, please reply 'No answer found'\n"
            #print(prompt)
            answer = ask_gpt(prompt)
            # Remove empty lines from the answer
            #answer = re.sub(r'\n\s*\n', '', answer)
            # Replace all newlines in the answer with spaces
            answer = answer.replace('\n', ' ')
            print(answer)
            # Add this answer to the dataframe
            papers_df.loc[papers_df['ArticleURL'] == url, question_string] = answer
            
    # Write out a copy of the papers_csv_file with the answers added, with the question string as the column name
    print(papers_df)
    # Write out the dataframe to a csv file in the same directory as the papers csv file
    print(f"Writing out to {full_file_name}")
    papers_df.to_csv(f"{full_file_name}", index=False)
    #papers_df.to_csv(f"{base_file_name}.answers.csv", index=False)
    
    #papers_df.to_csv(f"{papers_csv_file}_with_answers.csv", index=False)
        


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python answer_questions.py <papers_csv_file> <json_file> <question_string_or_file> [top_n_results=1] [min_sections=2]")
        sys.exit(1)
    papers_csv_file = sys.argv[1]
    json_file = sys.argv[2]
    question_string_or_file = sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    min_sections = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    main(papers_csv_file, json_file, question_string_or_file, n, min_sections)
    
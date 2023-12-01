#
# import openai
# from bs4 import BeautifulSoup
# import requests
#
# # Set your OpenAI API key
# openai.api_key = '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'
#
#
# # Function to extract job details from a website URL using OpenAI
# def extract_job_details(url):
#     # Perform web scraping to retrieve HTML content
#     response = requests.get(url)
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.content, 'html.parser')
#         webpage_text = soup.get_text()
#
#         # Use OpenAI's GPT-3 to analyze text and identify job-related content
#         # Modify the prompt and GPT-3 parameters according to your specific use case
#         prompt = f"Extract job details from: {url}\n\nWebpage content: {webpage_text}"
#
#         try:
#             completion = openai.Completion.create(
#                 engine="davinci",  # Adjust the engine according to your requirements
#                 prompt=prompt,
#                 max_tokens=150,  # Reduce the number of tokens for the completion
#                 n=1  # Number of completions to generate
#             )
#
#             if completion and len(completion.choices) > 0:
#                 extracted_text = completion.choices[0].text.strip()
#                 # For demonstration purposes, assuming extracted text is in JSON format
#                 job_details = extracted_text
#                 return job_details
#         except openai.error.OpenAIError as e:
#             print(f"OpenAI Error: {e}")
#
#     return None
#
#
# # Example URL to scrape job data from
# example_url = 'https://jobs.bdjobs.com/jobsearch.asp?fcatId=8&icatId='
#
# # Get job information from the URL
# job_info = extract_job_details(example_url)
#
# if job_info:
#     # Print the extracted job information
#     print(job_info)
#     # Here, you can further process or use the extracted job information as needed
# else:
#     print("Failed to extract job information.")

from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.llms import OpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import requests

# Define the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_splitter_step = RunnablePassthrough()

# Set your OpenAI API key
openai_api_key = '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'

# Define the OpenAI LLM
llm = OpenAI(api_key=openai_api_key, temperature=0.5)  # Adjust the temperature according to your requirements

# Define the output parser
output_parser = StrOutputParser()

# Define the Langchain pipeline
pipeline = RunnableMap(
    text=text_splitter_step,
    sliced_text=text_splitter,
    llm=llm,
    parsed_output=output_parser
)

# Function to extract job details from a website URL using Langchain
def extract_job_details(url):
    # Perform web scraping to retrieve HTML content
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        webpage_text = soup.get_text()

        # Pass the sliced text through the Langchain pipeline
        sliced_text = text_splitter.split_text(webpage_text)
        job_details = []

        for chunk in sliced_text:
            # Customize the prompt to extract job details
            prompt = f"Extract job details from the following text:\n\n{chunk}\n\n"

            # Generate the response using the OpenAI LLM
            response = llm.generate(prompt)

            # Parse the output to extract job details
            parsed_output = output_parser.parse(response)

            # Append the parsed job details to the list
            job_details.append(parsed_output)

        # Combine the parsed job details from each chunk
        combined_output = ''.join(job_details)

        return combined_output

    return None

# Example URL to scrape job data from
example_url = 'https://jobs.bdjobs.com/jobsearch.asp?fcatId=8&icatId='

# Get job information from the URL
job_info = extract_job_details(example_url)

if job_info:
    # Print the extracted job information
    print(job_info)
    # Here, you can further process or use the extracted job information as needed
else:
    print("Failed to extract job information.")
from PyPDF2 import PdfReader
import openai
import os
from os import listdir
from os.path import isfile, join
from CSVOutput import write_lists_to_csv


# absolute path of the directory which the pdfs are stored in
pdfs_directory = os.path.dirname(os.path.abspath(__file__)) + "/pdfs"

# absolute paths of each individual pdf
pdf_paths = [pdfs_directory + "/" + file for file in listdir(pdfs_directory) if
             file[-4:] == ".pdf" and isfile(join(pdfs_directory, file))]

openai.api_key = ""

prompts = [
    '''What devices were tested in this paper? Please give the items and a summary for each''',
    '''Can you briefly summarize the single event effect testing that was done and the
       found results for the device(s) that were tested?''',
    '''Can you briefly summarize the total ionizing dose testing that was done and the found results
       for the device(s) that were tested?''',
    '''Can you briefly summarize any interesting data that was found about the device
       that was tested in terms of radiation effects?'''
]

'''Converts a pdf into a string. Takes in pdf path as an argument'''


def pdfToString(path):
    text = ""

    with open(path, 'rb') as file:
        pdf = PdfReader(file)

        for page in pdf.pages:
            text += page.extract_text()
    return text


# runs an input through the gpt api and returns the output as a string
def gptInput(input):
    gpt = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": input}]
    )
    return gpt.choices[0].message.content


results = []

# iterate through each pdf path
for path_index, path in enumerate(pdf_paths):
    paper = pdfToString(path)
    # get the paper name
    index = path.find("/pdfs/")
    pdf_name = path[index+6:-4]

    results.append([])
    for prompt in prompts:
        full_prompt = prompt + paper

        step = int(len(paper) / 4)
        full_reply = ""

        paper_length = len(paper)

        paper_subsection = paper[0:int(paper_length / 3)]  # try different subsections of the paper
        full_prompt = prompt + paper_subsection

        full_reply += gptInput(full_prompt)
        results[path_index].append(full_reply)
    results[path_index].insert(0, pdf_name)

headers = ['Paper Name', 'Device Tested', 'Single Event Effects', 'Total Ionizing Dose', 'Interesting Data']
data = results
output_file = 'output.csv'
write_lists_to_csv(data, output_file, headers)
print(results)

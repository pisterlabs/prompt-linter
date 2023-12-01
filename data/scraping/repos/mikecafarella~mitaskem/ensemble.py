import argparse
import json
import os

from langchain.document_loaders import PyPDFLoader
from mitaskem.src.text_search import text_param_search, text_var_search

GPT_KEY = None 

def extract_paper_info(input_json_file, output_json_file):
    with open(input_json_file, 'r') as file:
        json_data = json.load(file)

    paper_info = []
    for entry in json_data:
        title = entry.get('title', 'N/A')
        doi = 'N/A'
        for identifier in entry.get('identifier', []):
            if identifier.get('type') == 'doi':
                doi = identifier.get('id')
                break
        url = 'N/A'
        for link in entry.get('link', []):
            if link.get('type') == 'publisher':
                url = link.get('url')
                break
        paper_info.append({'title': title, 'doi': doi, 'url': url})

    with open(output_json_file, 'w') as outfile:
        json.dump(paper_info, outfile, indent=4)

def load_paper_info(json_file):
    with open(json_file, 'r') as file:
        json_data = json.load(file)
    return json_data

def load_pdf(pdf_file, output_file):
    loader = PyPDFLoader(pdf_file)
    content = loader.load()
    content_str = '\n'.join(str(item) for item in content)  # Convert each element to a string and join them
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(content_str)

from mitaskem.src.methods import split_latex_into_chunks

# def async_extract_vars(in_path, out_dir):
#     out_filename_params = out_dir + "/" + in_path.split("/")[-1].split(".txt")[0] + "_params.txt"
#     out_filename_vars = out_dir + "/" + in_path.split("/")[-1].split(".txt")[0] + "_vars.txt"

#     with open(in_path, "r") as fi, open(out_filename_params, "w+") as fop, open(out_filename_vars, "w+") as fov:
#         text = fi.read()
#         chunks = split_latex_into_chunks(document=)

# @profile
def extract_vars(in_path, out_dir):
    out_filename_params = out_dir + "/" + in_path.split("/")[-1].split(".txt")[0] + "_params.txt"
    out_filename_vars = out_dir + "/" + in_path.split("/")[-1].split(".txt")[0] + "_vars.txt"

    with open(in_path, "r") as fi, open(out_filename_params, "w+") as fop, open(out_filename_vars, "w+") as fov:
        text = fi.read()
        length = len(text)
        segments = int(length / 3500 + 1)

        for i in range(segments):
            snippet = text[i * 3500: (i + 1) * 3500]

            # output, success = text_param_search(snippet, GPT_KEY)
            # if success:
            #     print("OUTPUT (params): " + output + "\n------\n")
            #     if output != "None":
            #         fop.write(output + "\n")

            output, success = text_var_search(snippet, GPT_KEY)
            if success:
                print("OUTPUT (vars): " + output + "\n------\n")
                if output != "None":
                    fov.write(output + "\n")

def extract_variables(text_file, output_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", type=str, default=text_file)
    parser.add_argument("-o", "--out_dir", type=str, default=output_dir)
    args = parser.parse_args()
    out_filename_params = args.out_dir + "/" + args.in_path.split("/")[-1].split(".txt")[0] + "_params.txt"
    out_filename_vars = args.out_dir + "/" + args.in_path.split("/")[-1].split(".txt")[0] + "_vars.txt"

    with open(args.in_path, "r") as fi, open(out_filename_params, "w+") as fop, open(out_filename_vars, "w+") as fov:
        text = fi.read()
        length = len(text)
        segments = int(length / 3500 + 1)

        for i in range(segments):
            snippet = text[i * 3500: (i + 1) * 3500]

            output, success = text_param_search(snippet, GPT_KEY)
            if success:
                print("OUTPUT (params): " + output + "\n------\n")
                if output != "None":
                    fop.write(output + "\n")

            output, success = text_var_search(snippet, GPT_KEY)
            if success:
                print("OUTPUT (vars): " + output + "\n------\n")
                if output != "None":
                    fov.write(output + "\n")

def emsemble(json_file, data_list):
    json_data = load_paper_info(json_file)


if __name__=="__main__":
    print("run main")
    # load_pdf("../../resources/xDD/paper/Time-Varying COVID-19 Reproduction Number in the United States.pdf",
    #          "../../resources/xDD/paper/Time-Varying COVID-19 Reproduction Number in the United States.txt")
    extract_variables("../../resources/xDD/paper/COVID-19 Vaccine Effectiveness by Product and Timing in New York State.txt", "/Users/chunwei/research/mitaskem/resources/xDD/params")
    # input_json_file = '../../resources/xDD/documents_mentioning_starterkit_data.bibjson'
    # output_json_file = '../../resources/xDD/xdd_paper.json'
    # extract_paper_info(input_json_file, output_json_file)

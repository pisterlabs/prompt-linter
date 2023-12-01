from src.context import PubMedScraper
from src.claude import Claude
from anthropic import HUMAN_PROMPT, AI_PROMPT

import time
import math
import argparse


def main():
    banner = r"""
   ___ _____ _   _ _____ ___ _   _ ____  
 / ___| ____| \ | | ____|_ _| | | / ___| 
| |  _|  _| |  \| |  _|  | || | | \___ \ 
| |_| | |___| |\  | |___ | || |_| |___) |
 \____|_____|_| \_|_____|___|\___/|____/ 
 O       o O       o O      o 0       o 0
| O   o | | O   o | | O   o | | 0   o | |
| | O | | | | O | | | | O | | | | 0 | | |
| o   O | | o   O | | o   O | | o   o | |
o       O o       O o       O o       0 o
 """

    intro = r"""
Welcome to Geneius - a tool for disease-gene evidence search and explanation. Geneius is a powerful command-line tool
leveraging Anthropic AI's Claude API to help you search through scientific literature and extract evidence for 
disease-gene associations. 

Please submit any issues or feature requests to our GitHub: https://github.com/aaronwtr/geneius/

For help and usage instructions, run 'geneius --help'.
"""
    print(banner)
    time.sleep(2)
    print(intro + "\n")

    parser = argparse.ArgumentParser(description="Geneius: A tool for disease-gene evidence search and explanation.")
    parser.add_argument("--disease", type=str, required=True, help="Disease name")
    parser.add_argument("--num_records", type=int, required=True, help="Number of records to search through")
    parser.add_argument("--api_key", type=str, required=True, help="Anthropic API key")
    parser.add_argument("--gene", type=str, help="Gene name (only for Task 1)")
    parser.add_argument("--num_genes", type=int, help="Number of genes (only for Task 2)")
    args = parser.parse_args()

    start_time = time.time()

    pms = PubMedScraper()
    disease = args.disease

    _claude = None
    prompt = None

    if args.gene:
        # Task 1
        gene = args.gene

        context, num_records = pms.get_literature_context(disease, args.num_records)
        claude = Claude(args.api_key)
        prompt = f"{HUMAN_PROMPT} Imagine you are an expert researcher going through the literature to extract " \
                 f"evidence implicating molecular involvement of gene {gene} in disease " \
                 f" {disease}. I want you to explain the molecular mechanism of the gene's involvement in " \
                 f"the disease based on the scientific context I am providing you. In order to " \
                 f"effectively retrieve information, I will provide you with context from scientific literature. You " \
                 f"can use your internal data and this context to formulate a response. If you are uncertain, do not " \
                 f"speculate. Restrict yourself to returning information confirming the connection of the disease  " \
                 f"and the gene, if there are any. Strictly return only papers that have a DOI available. Your  " \
                 f"response should look like <response>[Title]: 'paper title'\n [DOI]:'doi'\n [Explanation]: This " \
                 f"paper suggests [gene] is linked to [disease] [reason]</response> Take care to complete all " \
                 f"fields of your response entirely. \n\n  <context>{context}</context> {AI_PROMPT}"

    else:
        # Task 2
        num_genes = args.num_genes

        context = pms.get_literature_context(disease, args.num_records)
        claude = Claude(args.api_key)
        prompt = f"{HUMAN_PROMPT} Imagine you are an expert researcher going through the literature to find " \
                 f"{num_genes} genes that are involved in {disease}, and corresponding evidence implicating  " \
                 f"molecular involvement of the genes in disease {disease}. I want you to explain " \
                 f"the molecular mechanism of the gene's involvement in " \
                 f"the disease based on the scientific context I am providing you. In order to " \
                 f"effectively retrieve information, I will provide you with context from scientific literature. You " \
                 f"can use your internal data and this context to formulate a response. If you are uncertain, do not " \
                 f"speculate. Restrict yourself to returning information confirming the connection of the " \
                 f"disease and the gene, if there are any. Strictly only restrict to papers that have a DOI available" \
                 f". Your response  should look like <response>[Genes]: [Gene 1, Gene 2, ... Gene N] \n [Title]: " \
                 f"'paper title'\n [DOI]:'doi'\n [Explanation]: This paper suggests [gene] is linked to [disease] " \
                 f"[reason]</response> Take care to complete all fields of your response entirely. \n\n" \
                 f"<context>{context}</context> {AI_PROMPT}"

    claude.sync_stream(prompt)

    print(f"Collected and parsed through {args.num_records} scientific papers in: "
          f"{(math.floor((time.time() - start_time) / 60))} minutes and {math.floor((time.time() - start_time) % 60)} "
          f"seconds.")


# TODO:
#  - change entrypoint from geneius.__main__:main to __main__:main
#  - Redo upload

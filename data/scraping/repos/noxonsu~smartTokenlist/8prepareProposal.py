import json
import os
import requests
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from utils import *
#p8 - 8prepareProposal.py
#p6 - 6prepareSummary.py

CHAINBASE_API_URL= os.environ.get("CHAINBASE_API_URL")
NETWORK= os.environ.get("NETWORK")
MAINFILE = os.environ.get("MAINFILE")
print("mainfie"+MAINFILE)

def filter_sites_without_proposal(data):
    return [(entry['web_domains'][0], entry['contract_address']) for entry in data if not entry.get('p8') and entry.get('p6') and entry.get('tgGroupJoined') and entry.get('tgGroupJoined') == "success" and entry.get('web_domains')]


def load_summary(contract_address):
    filename = os.path.join('summaries', f"{contract_address}.json")
    with open(filename, 'r') as file:
        return json.load(file)['summary']

def load_system_message(filename):
    with open(filename, 'r') as file:
        return file.read()


def generate_message(targetSummary):
    chat = ChatOpenAI(temperature=0.5, model_name="gpt-4")
    system_message_content = load_system_message('GPT_proposal.txt')
    messages = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=f"Target project (this is not our project! analyse them to create an icebreaker and good offer): {targetSummary}")
    ]
    gpttitle = chat(messages)
    # Remove the quotation marks from the start and end of the generated title
    if gpttitle.content[0] == '"':
        gpttitle.content = gpttitle.content[1:]
    if gpttitle.content[-1] == '"':
        gpttitle.content = gpttitle.content[:-1]
    
    return gpttitle.content


def process_sites(data, sites_without_proposal):
    for domain, contract_address in sites_without_proposal:
        try:
            targetSummary = load_summary(contract_address)
        except:
            targetSummary = ""
        HOLDERS_COUNT = get_holders_count(contract_address)  # Get holders count
        # Concat "HOLDERSCOUNT" to targetSummary
        targetSummary = f"{targetSummary} . Amount of holders in {NETWORK} network: {HOLDERS_COUNT}"
        proposal = generate_message(targetSummary)
        if proposal:
            print(proposal)
            # Save the proposal to file
            proposal_filename = os.path.join('proposals', f"{contract_address}.json")
            with open(proposal_filename, 'w') as file:
                json.dump({'proposal': proposal}, file, indent=4)
            
            # Update the element in data
            for entry in data:
                if entry['contract_address'] == contract_address:
                    entry['p8'] = True
                    break  # Exit the loop once the matching entry is found and updated


def load_data_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def main():
    data = load_data_from_file(MAINFILE)  # load data from file
    sites_without_proposal = filter_sites_without_proposal(data)
    print(f"Found {len(sites_without_proposal)} sites without proposal")
    
    sites_without_proposal = sites_without_proposal[:10]

    if not sites_without_proposal:
        print("All sites processed")
    else:
        process_sites(data, sites_without_proposal)

        with open(MAINFILE, "w") as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()

from web3 import Web3
import os
import re
import json
from serpapi import GoogleSearch
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from utils import *

SERPAPI_KEY = os.environ.get('SERPAPI_KEY')
MAINFILE = os.environ.get("MAINFILE")
WEB3_RPC = os.environ.get("WEB3_RPC")
NETWORK = os.environ.get("NETWORK")
if not SERPAPI_KEY:
    print("Please set the SERPAPI_KEY environment variable.")
    exit()


def load_and_filter_contracts():
    with open(MAINFILE, 'r') as f:
        data = json.load(f)
        return data, [entry for entry in data if "web_domains" not in entry]

def findOfficialDomain(serp, project_name):
    
    chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    messages = [
        SystemMessage(content="Analyse SERP and find the official domain of the crypto token "+project_name+". we are only looking for new small projects. skip known domains of popular projects. Return only domain name if found. Return only domain name without quotes etc."),
        HumanMessage(content=f" {serp} \n\n The official domain is: ")
    ]

    try:
        response = chat(messages)
        gpttitle = response.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Not found"

    # Remove the quotation marks from the start and end of the generated title
    if gpttitle[0] == '"':
        gpttitle = gpttitle[1:]
    if gpttitle[-1] == '"':
        gpttitle = gpttitle[:-1]
    
    return gpttitle


def initialize_web3():
    w3 = Web3(Web3.HTTPProvider(WEB3_RPC))
    if not w3.is_connected():
        print("Not connected to Binance Smart Chain network!")
        exit()
    return w3

def load_abi(filename):
    with open(filename, "r") as abi_file:
        return json.load(abi_file)

def is_valid_domain(domain):
    pattern = r'^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]$'
    return bool(re.match(pattern, domain))
  
def search_google(nameOfProject):
    params = {
        "engine": "google",
        "q": nameOfProject,
        'gl': 'us',
        'hl': 'en',
        'num': 20,
        "api_key": SERPAPI_KEY,
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    # Error handling for missing key
    if "organic_results" in results:
        return results["organic_results"]
    else:
        print("Error: 'organic_results' not found in the results!")
        print(results)  # This will print the structure of results to inspect it
        return []


def load_scanned_contracts(filename='11scanned.txt'):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return set(line.strip().lower() for line in file)
    return set()


def save_scanned_contract(contract_address, filename='11scanned.txt'):
    with open(filename, 'a') as file:
        file.write(contract_address + '\n')


def main():
    data, filter_contracts = load_and_filter_contracts()
    scanned_contracts = load_scanned_contracts()
    
    filter_contracts = [contract for contract in filter_contracts if 
                        contract["contract_address"].lower() not in scanned_contracts and
                        contract.get('holders') and 
                        contract['holders'].get(NETWORK, float('inf')) < 300
                        ]
    
    print("11.py . contracts to scan:")
    print (len(filter_contracts))
    filter_contracts = filter_contracts[:30]
    
    

    w3 = initialize_web3()
    abi = load_abi("erc20.abi")
    for entry in filter_contracts:
        addr = Web3.to_checksum_address(entry["contract_address"])
        
        contract = w3.eth.contract(address=addr, abi=abi)
        name_project = " " + contract.functions.symbol().call() + " " + contract.functions.name().call()+"  -tokenview.io -t.me -youtube.com -facebook.com -github.com -beaconcha.in -abc.bi -medium.com -ethplorer.io -blockchair.com -coinmarketcap.com -site:binance.com -site:coinmarcetcap.com "
        print(name_project+"\n")
        organic_results = search_google(name_project)
        print(organic_results)
        
        serp=""
        for result in organic_results:
            #KeyError: 'snippet'
            if "snippet" not in result:
                result["snippet"] = ""
            serp += (str(result["position"]) +". " + result["link"]+" "+result["title"]+" "+result["snippet"])
        
        domain=findOfficialDomain(serp,name_project)
        print("domain: ")
        print(domain)
        save_scanned_contract(addr)
        # Check if a valid domain is found and save it to the original data
        if domain != "not found" and is_valid_domain(domain):
            # Check if domain already exists in the original data
            domain_exists = False
            for orig_entry in data:
                # Check if 'web_domains' key exists in orig_entry
                if 'web_domains' in orig_entry:
                    # Check if domain is in the list of domains for this entry
                    if domain in orig_entry["web_domains"]:
                        domain_exists = True
                        print("Domain already exists:", domain)  # Added print statement here
                        break
                        
            if not domain_exists:
                entry["web_domains"] = [domain]
                # Find the original entry in data and update it
                for orig_entry in data:
                    if orig_entry["contract_address"] == entry["contract_address"]:
                        orig_entry["web_domains"] = [domain]
                        break



    # Save the updated data back to the JSON file
    with open(MAINFILE, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    main()

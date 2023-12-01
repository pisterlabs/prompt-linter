# import libs
import json
import web3
import time
import requests
import openai
from io import BytesIO
from eth_abi.packed import encode_packed


# terminal 1: npx hardhat node
# terminal 2: npx hardhat run scripts/deploy.js --network localhost
# terminal 2: npx hardhat run scripts/request_callback.js --network localhost
# terminal 3: ipfs daemon
# terminal 4: python3 server/server.py



# connect to the local blockchain
web3 = web3.Web3(web3.HTTPProvider('http://localhost:8545'))

# load openAI API key from .oai_api
with open('.oai_api') as f:
    oai_api = f.read()
openai.api_key = oai_api

# load mnemonic from .eth_keys
with open('.eth_keys') as f:
    mnemonic = f.read()

web3.eth.account.enable_unaudited_hdwallet_features()

# create an account using the mnemonic
account = web3.eth.account.from_mnemonic(mnemonic)


# load libs/Oracle_Bot.json
oracle_bot_data = json.load(open('libs/Oracle_Bot.json'))

with open('libs/oracle_bot_abi.json') as f:
    oracle_bot_abi = json.load(f)

# create a contract instance using oracle_bot_data['abi'], oracle_bot_data['address']
oracle_bot = web3.eth.contract(address=oracle_bot_data['address'], abi=oracle_bot_abi)

# load last_checked_block.txt create if not exists
try:
    with open('last_checked_block.txt') as f:
        last_checked_block = int(f.read())
except:
    last_checked_block = 0
    # create last_checked_block.txt
    with open('last_checked_block.txt', 'w') as f:
        f.write(str(last_checked_block))

# load known_ids.json, create if not exists
try:
    known_ids = json.load(open('known_ids.json'))
except:
    known_ids = []

    with open('known_ids.json', 'w') as f:
        json.dump(known_ids, f)

bot_model = "gpt-3.5-turbo"

def upload_to_ipfs(doc_to_store, ipfs_url):
    # expected url is like http://localhost:5001
    # Python doesn't have a Blob type like JavaScript, so we'll use BytesIO
    blob = BytesIO(doc_to_store.encode('utf-8'))
    
    # Creating the FormData
    files = {'file': blob}
    
    # Making the first request
    response = requests.post(ipfs_url + '/api/v0/add', files=files)
        
    result = response.json()
    cid = result['Hash']

    # Pin the cid
    response_pin = requests.post(ipfs_url + '/api/v0/pin/add?arg=' + cid)

    return cid


def process_document_request(document_request):
    # filter document_request['request'] for keywords and appropriate content
    # build out prompt around request
    prompt_prefix = "This is a request from a user for you to answer:\n\n"
    prompt_suffix = "\n\nPlease respond to this request by adding your response below:\n\n"

    prompt = prompt_prefix + document_request['request'] + prompt_suffix
    # run prompt through LLM
    bot_response = openai.ChatCompletion.create(
        model=bot_model,
        n = 1,
        messages=[
            {
                "role": "user", 
                "content": prompt
            },
            {
                "role": "system", 
                "content": "your job is to answer the user's request"
            }
        ]
    )

    # format response
    formatted_response = bot_response.choices[0].message.content
    
    # store response in IPFS
    cid = upload_to_ipfs(formatted_response, 'http://localhost:5001')
    return cid

def process_callback_request(callback_request):
    # filter document_request['request'] for keywords and appropriate content



    # TODO if parsing fails run through LLM again (assuming the LLM uses a random seed)

    print("callback_request:")
    print(callback_request)
    # parse the data types from the callback_signature
    data_types = callback_request["callback_signature"].split('(')[1].split(')')[0].split(',')

    # document_request['param_names'] = "'document_idx','document_cid'"
    # parse the param_names from the param_names
    param_names = callback_request["param_names"].split(',')

    # build out prompt around request
    prompt_prefix = "This is a request from a user for you to answer:\n\n"
    prompt_suffix = "\n\nPlease respond to this request in the following json format: \n\n"
    data_type_array = []
    json_object_string = "{"
    for i in range(len(param_names)):
        json_object_string += "\n\t" + param_names[i] + ": \"" + data_types[i] + "\","
        data_type_array.append(data_types[i])
    json_object_string = json_object_string[:-1]
    json_object_string += "}"

    prompt = prompt_prefix + callback_request['request'] + prompt_suffix + json_object_string + "\n\n Please keep strings down to 100 characters or less."

    # run prompt through LLM
    bot_response = openai.ChatCompletion.create(
        model=bot_model,
        n = 1,
        messages=[
            {
                "role": "user", 
                "content": prompt
            },
            {
                "role": "system", 
                "content": "your job is to answer the user's request"
            }
        ]
    )

    # formated_response: parse out json object from bot_response.choices[0].message.content from the rest of the text starting with the first '{' and ending with the last '}'
    formatted_response = "{" + bot_response.choices[0].message.content.split('{')[1].split('}')[0] + "}"

    print(formatted_response)
    # TODO: validate formatted response for solidity data type in the json object, should support string, uint256, address, bool, bytes32, bytes, uint32, uint16 etc.
    for data_type in data_type_array:
        if data_type == "uint256":
            try:
                int(formatted_response[param_names[data_type_array.index(data_type)]])
            except:
                print("invalid uint256")
                return False
        elif data_type == "address":
            try:
                web3.toChecksumAddress(formatted_response[param_names[data_type_array.index(data_type)]])
            except:
                print("invalid address")
                return False
        elif data_type == "bool":
            try:
                bool(formatted_response[param_names[data_type_array.index(data_type)]])
            except:
                print("invalid bool")
                return False
        elif data_type == "bytes32":
            try:
                bytes.fromhex(formatted_response[param_names[data_type_array.index(data_type)]])
            except:
                print("invalid bytes32")
                return False



    return formatted_response, json_object_string, data_type_array


# once per interval, check if there are any new Document_Requested events on the Oracle_Bot smart contract
interval = 10 # seconds
while True:
    # get the latest block
    latest_block = web3.eth.block_number 

    if(latest_block <= last_checked_block):
        print("no new blocks")
        time.sleep(interval)
        continue
    print("checking from block {} to block {}".format(last_checked_block, latest_block))
    # get the doc_request_events from the last_checked_block to the latest_block
    doc_request_events = oracle_bot.events.Document_Requested.get_logs(fromBlock=last_checked_block+1, toBlock=latest_block)
    callback_request_events = oracle_bot.events.Callback_Requested.get_logs(fromBlock=last_checked_block+1, toBlock=latest_block)
    
    # for each event, print the event data
    for doc_event in doc_request_events:  
        id = doc_event['args']['document_idx']
        
        # get CID from the contract oracle_bot.get_content_cid
        cid = oracle_bot.functions.get_content_cid(id).call()
        
        # pull file from local IPFS node
        r = requests.get('http://localhost:8080/ipfs/{}'.format(cid))

        document_request = {'id': id, 'cid': cid, 'request': r.text}
        # add to known ids
        known_ids.append(document_request)
        
        response_cid = process_document_request(document_request)

        print("response cid: {}".format(response_cid))
        
        # store IPFS CID in Oracle_Bot smart contract as response using account
        oracle_bot.functions.fulfill_document(document_request['id'], cid).transact()

    for callback_event in callback_request_events:
        id = callback_event['args']['document_idx']
        
        # get CID from the contract oracle_bot.get_content_cid
        cid = oracle_bot.functions.get_content_cid(id).call()

        callback_request = oracle_bot.functions.callback_requests(id).call()

        
        # pull file from local IPFS node
        r = requests.get('http://localhost:8080/ipfs/{}'.format(cid))

        callback_request = {
            'id': id, 
            'cid': cid, 
            'request': r.text, 
            'callback_signature': callback_request[1], 
            'param_names': callback_request[2]
        }
        # add to known ids
        known_ids.append(callback_request)
        
        response, json_object_string, data_type_array = process_callback_request(callback_request)

        # create json object from response and json_object_string
        response_object = json.loads(response)
        # TODO insure that the response_object matches the format of the json_object_string
        
        # iterate over the response_object and make an array of the values
        response_array = []

        # iterate over response_object dict
        # at the same time iterate over data_type_array
        # if the data_type is uint256, convert the value to int
        # if the data_type is string, leave it as is
        # if the data_type is bytes32, convert the value to bytes
        # if the data_type is bytes, convert the value to bytes
        # if the data_type is bool, convert the value to bool
        # if the data_type is address, convert the value to address
        idx = 0
        for key in response_object:
            print(key)
            if data_type_array[idx] == 'uint256':
                print("response_object[key]: {}".format(response_object[key]))
                response_array.append(int(response_object[key]))
            elif data_type_array[idx] == 'string':
                response_array.append(response_object[key])
            elif data_type_array[idx] == 'bytes32':
                response_array.append(bytes(response_object[key], 'utf-8'))
            elif data_type_array[idx] == 'bytes':
                response_array.append(bytes(response_object[key], 'utf-8'))
            elif data_type_array[idx] == 'bool':
                response_array.append(bool(response_object[key]))
            elif data_type_array[idx] == 'address':
                response_array.append(response_object[key])
            idx += 1

        print("response array: {}".format(response_array))
        print("data type array: {}".format(data_type_array))

        # abi encode the response_array
        # response = web3.eth.abi.encode_abi(data_type_array, response_array)
        
        response = encode_packed(data_type_array, response_array)      
        print("response: {}".format(response.hex()))  
        
        # store IPFS CID in Oracle_Bot smart contract as response using account
        oracle_bot.functions.fulfill_callback(callback_request['id'], "0x" +response.hex()).transact()

    # update last_checked_block.txt
    with open('last_checked_block.txt', 'w') as f:
        f.write(str(latest_block))

    # update and save known_ids.json
    with open('known_ids.json', 'w') as f:
        json.dump(known_ids, f)

    last_checked_block = latest_block
    # wait 60 seconds
    time.sleep(interval)

    
import requests
import time
import json
import os
import sys
import openai
from itertools import islice
from stellar_sdk import Keypair, Network, Server, TransactionBuilder, Asset, Account


if os.name=='nt':
    os.system("cls")
if os.name=='posix':
    os.system("clear")



global FUTURENET_URL;FUTURENET_URL='https://horizon-futurenet.stellar.org'
FUTURENET_SERVER=Server(FUTURENET_URL)
global NEURALSHARE_ACCOUNT;NEURALSHARE_ACCOUNT='GAU3XW7Z5OYR7BFANSB7HGVGJRZJ5NSJ3ERWJ5HRVZRCHFSRTQ32YPEM'
global JSON_CONFIG;JSON_CONFIG={}
def new_keypair(airdrop=False):
    keypair = Keypair.random()
    requests.get('https://friendbot-futurenet.stellar.org', params={"addr": keypair.public_key})
    return {'secret':keypair.secret, 'address':keypair.public_key}
def new_configuration():
    return {'openai_apikey':None,"stellar_futurenet_secret":new_keypair()['secret']}
if os.path.isfile("config.json"):
    f=open("config.json","r")
    j=json.load(f)
    JSON_CONFIG=j
    
    
else:
    print("No configuration found in this folder. Creating a new one...\n\n")
    f=open("config.json","w")
    json.dump(new_configuration(),f,indent=4)
    f.close()
    print("Created new config.json in "+os.getcwd()+"\n\n")
    sys.exit()
def get_last_transaction():
    server = Server(horizon_url="https://horizon-futurenet.stellar.org")
    account_id = "GAU3XW7Z5OYR7BFANSB7HGVGJRZJ5NSJ3ERWJ5HRVZRCHFSRTQ32YPEM"
    last_cursor = "now" 
    for tx in server.payments().for_account(account_id).cursor(last_cursor).stream():
        if tx['asset_code']=='NEUR' and tx['asset_issuer']=='GAIDLQERTCWHVEY7QXQYSRIEZSCH3EQYP5RTYDT44POFVSKMCIOWUUJP' and tx['from']!='GAU3XW7Z5OYR7BFANSB7HGVGJRZJ5NSJ3ERWJ5HRVZRCHFSRTQ32YPEM' and tx['transaction_successful']==True:
           r=requests.get('https://horizon-futurenet.stellar.org/transactions/'+tx['transaction_hash']).json()
           txmemotype=r['memo_type']
           txsender=r['source_account']
           if txmemotype=='text':
            return r['memo'],txsender,tx['id']


#if prompt or response length is longer than Stellar's text memo max length (28 bytes) it will be splitted in chunks
def chunk(prompt, size=28):
    prompt=list(prompt)
    if len(prompt)>size:
        it = iter(prompt)
        itt=list(iter(lambda: tuple(islice(it, size)), ()))
        prompt= [''.join(tups) for tups in itt]
        return prompt
    else:
        prompt=''.join(prompt)
        return [prompt]

        
def send_transaction(sender_secret,dest_address,message,assettype):
    PASSPHRASE='Test SDF Future Network ; October 2022'
    #assettype=Asset("NEUR","GAIDLQERTCWHVEY7QXQYSRIEZSCH3EQYP5RTYDT44POFVSKMCIOWUUJP")
    sender_public_key=Keypair.from_secret(sender_secret).public_key
    
    server=FUTURENET_SERVER
    transaction = (
        TransactionBuilder(

            source_account=server.load_account(sender_public_key),
            network_passphrase=PASSPHRASE,
            base_fee=100,
                        
        )
        .append_payment_op(
            destination=dest_address, amount="0.0000001", asset=assettype
        )
        .add_text_memo(message)
        .set_timeout(30)
        .build()
    )
    transaction.sign(sender_secret)
    response = server.submit_transaction(transaction)

    return response
def get_last_10_tx_memos():
    account='GAU3XW7Z5OYR7BFANSB7HGVGJRZJ5NSJ3ERWJ5HRVZRCHFSRTQ32YPEM'
    server = Server(horizon_url="https://horizon-futurenet.stellar.org")

    transactions = server.transactions().for_account(account_id=account).order(desc=True).call()

    l=[]

        
        
    for x in range(len(transactions['_embedded']['records'])):
        l.append(transactions['_embedded']['records'][x]['memo'])
    return l


def gpt3_response(prompt):
    openai.api_key=JSON_CONFIG['openai_apikey']
    try:

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt="Human: "+prompt+"\nAI:\n",
            temperature=1,
            max_tokens=4000,
            top_p=1,
            
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=[" Human:", " AI:"],
            
        )
    except:
        return 0
    else:
        return response

def send_response(prompt, destination,txid):

    OPEN="<"+str(txid)+">"
    
    CLOSE="</"+str(txid)+">"
    


    last_10_memos=get_last_10_tx_memos()
    
    if OPEN in last_10_memos or CLOSE in last_10_memos:
        pass 
    else:
        resp=gpt3_response(prompt)[0]
        
        if resp==0:
            print("Error get response from Openai. Make sure your apikey is valid")
        else:
            
            resp=chunk(resp)
            print(resp)     
            send_transaction(JSON_CONFIG['stellar_futurenet_secret'],destination,CLOSE,Asset.native())
            for x in resp:
                send_transaction(JSON_CONFIG['stellar_futurenet_secret'],destination,x,Asset.native())
            send_transaction(JSON_CONFIG['stellar_futurenet_secret'],destination,OPEN,Asset.native())

            print("...Response Sent...")

            

    









    



print(":: ...LISTENING... ::")


LAST_TX_ID=''
while True:
      t=get_last_transaction()
      v=str(t[2])
      
      
      if LAST_TX_ID!=v:
         tm=datetime.datetime.now()
         print("______________")
         print(f'|{tm}|')
         print("...New Request... ")

         LAST_TX_ID=v
         send_response(t[0],t[1],t[2])
      



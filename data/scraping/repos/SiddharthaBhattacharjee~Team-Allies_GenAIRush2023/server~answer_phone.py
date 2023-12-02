from flask import Flask, request
import requests
import time
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from threading import Thread
from gpt import send_req
from datetime import datetime
import json
from sendSMS import sendMSG
from web3 import Web3
from env import getPrivateKey
import random

app = Flask(__name__)
base_url = "https://api.assemblyai.com/v2"

headers = {"authorization": "56b358be260841d0acc6c24xxxxxxx"}
client = Client("ACb8b05c3a3b447064a4dc61795xxxxxxx",
                "7729c783fd989ba8e62c04bac6xxxxxx")

#Eth Starts Here

#Eth Ends Here


@app.route("/")
def hello():
  return "Hello World!"


@app.route("/answer", methods=['GET', 'POST'])
def answer_call():
  """Respond to incoming phone calls with a brief message."""
  # Start our TwiML response
  resp = VoiceResponse()

  # Read a message aloud to the caller
  resp.say(
    "Hello, this is AI the emergency helpline. Please state your name, current location, and describe your emergency situation, Cut the call when you are finished, Your request will be registered once you end the call and we will notify you as your request is registered.",
    # "hi",
    voice='alice')
  resp.record(maxLength="120", action="/handle-recording")
  resp.say("Thank you for your message. Goodbye.", voice='alice')
  return str(resp)


@app.route("/handle-recording", methods=['GET', 'POST'])
def handle_recording():
  recording_url = request.form['RecordingUrl']
  data = {"audio_url": recording_url}
  url = base_url + "/transcript"
  response = requests.post(url, json=data, headers=headers)
  transcript_id = response.json()['id']
  polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
  while True:
    transcription_result = requests.get(polling_endpoint,
                                        headers=headers).json()

    if transcription_result['status'] == 'completed':
      # Get the call sid from the request
      call_sid = request.form['CallSid']

      # Get the call details from Twilio
      call = client.calls(call_sid).fetch()
      caller_number = call.from_formatted

      print(caller_number)
      print("Getting priority info from OpenAI")
      # transcription_text = transcription_result['text']
      # transcription_text = transcription_text.replace('\'', '')
      transcription_text = transcription_result.get('text', '')
      print(f'transcription_result: {transcription_result}')
      print(f'transcription_text before: {transcription_text}')
      transcription_text = transcription_text.replace('\'', '')
      print(f'transcription_text after: {transcription_text}')

      resp = send_req(transcribed_text=transcription_text)
      #   summary = transcription_text[:150] + "..."
      print("Got response from OpenAI")
      msg = "";
      name="";
      try:
        print("Debug:: TRY")
        print(resp)
        d = json.loads(resp)
        print(d["emergency_type"])
        msg = d["reply_msg"]
        name = d["name"]
        location = d["location"]
        emergency_type = d["emergency_type"]
        priority = d["priority"]
      except:
        print("Debug:: EXCEPT")

      print("Sending message to user")

      #   print(message.status_code)
      rand_12_digit_id = random.randint(100000000000, 999999999999)
      getCurrentTime = str(datetime.now())
      print(name, location, emergency_type, transcription_text, priority,
            getCurrentTime, rand_12_digit_id)
      msg = msg.replace('\'', '')
      print(f'msg after: {msg}')
      data = {
        "name": name,
        "Location": location,
        "priority": priority,
        "attention": emergency_type,
        "summary": msg,
        "phone": caller_number,
        "transcription": transcription_text,
        "time": getCurrentTime
      }
      s = json.dumps(data)
      #eth starts here
      provider = Web3.HTTPProvider(
        'https://eth-sepolia.g.alchemy.com/v2/zDdlaNQNTz6PrbF8nXtdvbM6BEt2aMFW'
      )
      w3 = Web3(provider)

      contract_address = Web3.to_checksum_address(
        "0x4d0A6fdeAF2A899895ad79C326D4777a5502D18E")

      contract_abi = [
    {
      "inputs": [],
      "stateMutability": "nonpayable",
      "type": "constructor"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "id",
          "type": "uint256"
        },
        {
          "internalType": "string",
          "name": "D",
          "type": "string"
        }
      ],
      "name": "addOperation",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "op",
          "type": "address"
        }
      ],
      "name": "addToWhitelist",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "id",
          "type": "uint256"
        }
      ],
      "name": "delReq",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "getOperations",
      "outputs": [
        {
          "components": [
            {
              "internalType": "uint256",
              "name": "id",
              "type": "uint256"
            },
            {
              "internalType": "string",
              "name": "data",
              "type": "string"
            },
            {
              "internalType": "uint256",
              "name": "status",
              "type": "uint256"
            }
          ],
          "internalType": "struct Contract.OperationData[]",
          "name": "",
          "type": "tuple[]"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "un",
          "type": "uint256"
        }
      ],
      "name": "getSOSNum",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "isWhiteListed",
      "outputs": [
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "op",
          "type": "address"
        }
      ],
      "name": "removeFromWhitelist",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "un",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "sn",
          "type": "uint256"
        }
      ],
      "name": "setSOSNum",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "id",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "s",
          "type": "uint256"
        }
      ],
      "name": "setStatus",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    }
  ]

      contract = w3.eth.contract(address=contract_address, abi=contract_abi)
      sender_address = Web3.to_checksum_address(
        "0x8453ada3A9E671E0f115B2f2A2939b03aD519615")
      w3.eth.defaultAccount = sender_address
      nonce = w3.eth.get_transaction_count(sender_address)
      data = contract.functions.getOperations().build_transaction({
        'gas':
        100000,
        'gasPrice':
        w3.to_wei('20', 'gwei'),
        'nonce':
        nonce,
      })
      signed_txn = w3.eth.account.sign_transaction(data,
                                                   private_key=getPrivateKey())
      tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
      tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
      try:
        transaction = contract.functions.addOperation(
          rand_12_digit_id, s).build_transaction({
            'gas':
            2000000,
            'gasPrice':
            w3.to_wei('50', 'gwei'),
            'nonce':
            w3.eth.get_transaction_count(sender_address),
          })

        # Sign and send the transaction
        signed_txn = w3.eth.account.sign_transaction(
          transaction, private_key=getPrivateKey())
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)

        print(tx_hash)
        print("Transaction Completed Successfully.")
        print("\n")
      except Exception as e:
        print("Transaction Failed!", e)
        print("\n")
        #eth ends here
      print(sendMSG(msg, caller_number))
      #   #Write data in blockchain
      break

    elif transcription_result['status'] == 'error':
      raise RuntimeError(
        f"Transcription failed: {transcription_result['error']}")

    else:
      time.sleep(3)
  return "OK"


def run():
  app.run(host='0.0.0.0', port=80)


def keep_alive():
  t = Thread(target=run)
  t.start()


keep_alive()

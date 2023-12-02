from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import re
import requests
import ast
import subprocess
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from substrateinterface import SubstrateInterface, Keypair
from substrateinterface.contracts import ContractCode, ContractInstance
from substrateinterface.exceptions import SubstrateRequestException
import os
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv


substrate_relay = SubstrateInterface(url="wss://shibuya-rpc.dwellir.com")
base_url = "https://shibuya.api.subscan.io"

mnemonic = ''
openai = ''
api_key = os.getenv("API_KEY")

def format_balance(amount: int):
    amount = format(
        amount / 10**substrate_relay.properties.get('tokenDecimals', 0),
        ".15g")
    return f"{amount} {substrate_relay.properties.get('tokenSymbol', 'UNIT')}"

def get_account_transfers(account_address):
    url = base_url + "/api/scan/transfers"
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    payload = {"address": account_address, "row": 100}

    response = requests.post(url, headers=headers, json=payload)
    return response.json()

def get_account_balance(account_address):
    result = substrate_relay.query("System", "Account", [account_address])
    balance = (result.value["data"]["free"] + result.value["data"]["reserved"])

    return format_balance(balance)


def get_polkadot_account_balance(account_address):
    result = substrate_relay.query("System", "Account", [account_address])
    balance = (result.value["data"]["free"] + result.value["data"]["reserved"])

    return format_balance(balance)


def send_balance(recipient_address, amount):
    call = substrate_relay.compose_call(call_module='Balances',
                                        call_function='transfer',
                                        call_params={
                                            'dest': recipient_address,
                                            'value': amount * 10**18
                                        })
    load_dotenv()
    mnemonic = os.getenv("MNEMONIC")
    print(mnemonic)
    extrinsic = substrate_relay.create_signed_extrinsic(
        call=call,
        keypair=Keypair.create_from_mnemonic(mnemonic),
        era={'period': 64})

    try:
        receipt = substrate_relay.submit_extrinsic(extrinsic,
                                                   wait_for_inclusion=True)

        print('Extrinsic "{}" included in block "{}"'.format(
            receipt.extrinsic_hash, receipt.block_hash))

        print(receipt)

        if receipt.is_success:
            print('✅ Success, triggered events:')
            for event in receipt.triggered_events:
                print(f'* {event.value}')
        else:
            print('⚠️ Extrinsic Failed: ', receipt.error_message)

        return receipt

    except Substrate_relayRequestException as e:
        print(e)
        return False


def get_transfer_details(extrinsic_index):
    url = base_url + "/api/scan/extrinsic"
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    payload = {"extrinsic_index": extrinsic_index}

    response = requests.post(url, headers=headers, json=payload)
    data =  response.json()
    res_list = []

    # Get all the transfers
    transfers = data['data']['transfers']

    # Get the first 3 and last 3 transfers
    selected_transfers = transfers[:3] + transfers[-3:]
    
    for transfer in selected_transfers:

        # Get timestamp and amount
        timestamp = transfer['block_timestamp']
        amount = transfer['amount']

        # Get addresses and truncate
        from_address = transfer['from'][:5] + '...' + transfer['from'][-5:]
        to_address = transfer['to'][:5] + '...' + transfer['to'][-5:]

        res_list.append((f"Timestamp: {timestamp}, Amount: {amount}, From: {from_address}, To: {to_address}"))
    return res_list


def get_erc20_total_supply(contract_address):
    load_dotenv()
    mnemonic = os.getenv("MNEMONIC")
    print(mnemonic)
    contract = ContractInstance.create_from_address(
        contract_address=contract_address,
        metadata_file=os.path.join(os.getcwd(), '../assets', 'erc20.json'),
        substrate=substrate_relay)
    result = contract.read(Keypair.create_from_mnemonic(mnemonic),
                           'total_supply')

    return str(result['result'])


def get_erc20_of_user(contract_address, user_address):
    load_dotenv()
    mnemonic = os.getenv("MNEMONIC")
    print(mnemonic)
    contract = ContractInstance.create_from_address(
        contract_address=contract_address,
        metadata_file=os.path.join(os.getcwd(), '../assets', 'erc20.json'),
        substrate=substrate_relay)
    result = contract.read(Keypair.create_from_mnemonic(mnemonic),
                           'balance_of',
                           args={'owner': user_address})
    return str(result['result'])


def transfer_erc20_to_user(contract_address, user_address, value):
    load_dotenv()
    mnemonic = os.getenv("MNEMONIC")
    print(mnemonic)
    contract = ContractInstance.create_from_address(
        contract_address=contract_address,
        metadata_file=os.path.join(os.getcwd(), '../assets', 'erc20.json'),
        substrate=substrate_relay)
    gas_predit_result = contract.read(
        Keypair.create_from_mnemonic(mnemonic),
        'transfer',
        args={
            'to': user_address,
            'value': value
        },
    )
    result = contract.exec(Keypair.create_from_mnemonic(mnemonic),
                           'transfer',
                           args={
                               'to': user_address,
                               'value': value
                           },
                           gas_limit=gas_predit_result.gas_required)

    return f"Transaction Hash: {result.extrinsic_hash}"


def filter(x):
    # filter based on source code
    if "something" in x["text"].data()["value"]:
        return False

    # filter based on path e.g. extension
    metadata = x["metadata"].data()["value"]
    return "only_this" in metadata["source"] or "also_that" in metadata[
        "source"]



class GetAccountBalanceInput(BaseModel):
    """Inputs for get_account_balance"""

    account_address: str = Field(
        description="the address of the account to get the balance of")


class GetAccountBalanceTool(BaseTool):
    name = "get_account_balance"
    description = """
        Useful when you want to get the balance of an polkadot account.
        The account address is the address of the account you want to get the balance of.
        The address format is ss58 encoded.
        """
    args_schema: Type[BaseModel] = GetAccountBalanceInput

    def _run(self, account_address: str):
        account_balance = get_account_balance(account_address)
        return account_balance

    def _arun(self, account_address: str):
        raise NotImplementedError(
            "get_current_stock_price does not support async")


class SendSubstrateBalanceInput(BaseModel):
    """Inputs for send_substrate_balance"""

    recipient_address: str = Field(
        description="the address of the account to send the balance to")
    amount: float = Field(description="the amount to send.")


class SendSubstrateBalanceTool(BaseTool):
    name = "send_substrate_balance"
    description = """
        Useful when you want to send a balance to a polkadot account.
        If balance is not specified send 0.001
        We will be sending Shibuya Testnet tokens/SBY.
        returns the extrinsic hash if successful
        """
    args_schema: Type[BaseModel] = SendSubstrateBalanceInput

    def _run(self, recipient_address: str, amount: int):
        res = send_balance(recipient_address, amount)
        return res.extrinsic_hash

    def _arun(self, account_address: str):
        raise NotImplementedError(
            "get_current_stock_price does not support async")


class ListAllTransactionsInput(BaseModel):
    """Inputs for list_all_transactions"""

    account_address: str = Field(
        description="the address of the account to get the transactions of")


class ListAllTransactionsTool(BaseTool):
    name = "list_all_transactions"
    description = """
        Useful when you want to list all transactions of a polkadot account.
        Lists the last first 3 and last 3 transactions.
        """
    args_schema: Type[BaseModel] = ListAllTransactionsInput

    def _run(self, account_address: str):
        res = get_account_transfers(account_address)
        return res

    def _arun(self, account_address: str):
        raise NotImplementedError(
            "list_all_transactions does not support async")


class GetTransferDetailsInput(BaseModel):
    """Inputs for get_transfer_details"""

    extrinsic_hash: str = Field(
        description=
        "the extrinsic hash of the transaction to get the details of, starts with 0x"
    )


class GetTransferDetailsTool(BaseTool):
    name = "get_transfer_details"
    description = """
        Useful when you want to get the details of a transaction.
        returns code, if successful, block time and data if it exists.
        """
    args_schema: Type[BaseModel] = GetTransferDetailsInput

    def _run(self, extrinsic_hash: str):
        res = get_transfer_details(extrinsic_hash)
        return ["successfully retreived data. Data:", res]

    def _arun(self, account_address: str):
        raise NotImplementedError(
            "get_transfer_details does not support async")


class GetERC20TotalSupplyInput(BaseModel):
    """Inputs for get_erc20_total_supply"""

    contract_address: str = Field(
        description="the address of the contract to get the total supply of")


class GetERC20TotalSupplyTool(BaseTool):
    name = "get_erc20_total_supply"
    description = """
        Useful when you want to get the total supply of an ERC20 token.
        The address of the contract should be given
        returns the total supply of the ERC20 token.
        """
    args_schema: Type[BaseModel] = GetERC20TotalSupplyInput

    def _run(self, contract_address: str):
        res = get_erc20_total_supply(contract_address)
        return res

    def _arun(self, account_address: str):
        raise NotImplementedError(
            "get_erc20_total_supply does not support async")


class GetERC20OfUserInput(BaseModel):
    """Inputs for get_erc20_of_user"""

    contract_address: str = Field(
        description="the address of the contract to get the balance of")
    user_address: str = Field(
        description="the address of the user to get the balance of")


class GetERC20OfUserTool(BaseTool):
    name = "get_erc20_of_user"
    description = """
        Useful when you want to get the balance of an ERC20 token of a user when given user address.
        The address of the contract should be given
        returns the balance of the ERC20 token of the user.
        """
    args_schema: Type[BaseModel] = GetERC20OfUserInput

    def _run(self, contract_address: str, user_address: str):
        res = get_erc20_of_user(contract_address, user_address)
        return res

    def _arun(self, account_address: str):
        raise NotImplementedError("get_erc20_of_user does not support async")


class TransferERC20ToUserInput(BaseModel):
    """Inputs for transfer_erc20"""

    contract_address: str = Field(
        description="the address of the contract to transfer the balance of")
    user_address: str = Field(
        description="the address of the user to transfer the balance of")
    amount: int = Field(description="the amount to transfer")


class TransferERC20ToUserTool(BaseTool):
    name = "transfer_erc20_to_user"
    description = """
        Useful when you want to transfer an ERC20 token to a user for a given amount.
        The address of the contract and amount should be given
        returns the transaction hash.
        """
    args_schema: Type[BaseModel] = TransferERC20ToUserInput

    def _run(self, contract_address: str, user_address: str, amount: int):
        res = transfer_erc20_to_user(contract_address, user_address, amount)
        return res

    def _arun(self, account_address: str):
        raise NotImplementedError("transfer_erc20 does not support async")
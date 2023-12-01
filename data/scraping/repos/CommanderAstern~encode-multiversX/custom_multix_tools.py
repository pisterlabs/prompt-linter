from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import os
from multiversx_sdk_core import Address
from multiversx_sdk_network_providers import ProxyNetworkProvider
from dotenv import load_dotenv

provider = ProxyNetworkProvider("https://devnet-gateway.multiversx.com")

load_dotenv()
mnemonic = ""
openai = ""
api_key = os.getenv("API_KEY")


def get_account_balance(account_address):
    acc = Address.from_bech32(account_address)
    account_on_network = provider.get_account(acc)
    return account_on_network.balance


class GetAccountBalanceInput(BaseModel):
    """Inputs for get_account_balance"""

    account_address: str = Field(
        description="the address of the account to get the balance of"
    )


class GetAccountBalanceTool(BaseTool):
    name = "get_account_balance"
    description = """
        Useful when you want to get the balance of an multiversx account.
        The account address is the address of the account you want to get the balance of.
        The address should start with a prefix erd
        """
    args_schema: Type[BaseModel] = GetAccountBalanceInput

    def _run(self, account_address: str):
        account_balance = get_account_balance(account_address)
        return account_balance

    def _arun(self, account_address: str):
        raise NotImplementedError("get_account_balance does not support async")


# Send Transaction


class SendTransactionInput(BaseModel):
    """Inputs for send_transaction"""

    receiver_address: str = Field(
        description="the address of the account to send the transaction to"
    )
    value: int = Field(description="the amount of eGold to send")


class SendTransactionTool(BaseTool):
    name = "send_transaction"
    description = """
        Useful when you want to send a transaction from an multiversx account.
        Returns the receiver address and the value of the transaction.

        The account address is the address of the account you want to send the transaction from.
        The receiver address is the address of the account you want to send the transaction to.
        The value is the amount of eGold you want to send.
        The address should start with a prefix erd
        """
    args_schema: Type[BaseModel] = SendTransactionInput

    def _run(self, receiver_address: str, value: int):
        if len(receiver_address) != 62:
            return "Invalid receiver address"

        # sendstr = f"$start${receiver_address}|{value}$end$"
        # return sendstr
        return (receiver_address, value)

    def _arun(self, receiver_address: str, value: int):
        raise NotImplementedError("send_transaction does not support async")

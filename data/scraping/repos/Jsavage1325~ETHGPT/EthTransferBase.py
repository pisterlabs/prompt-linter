from langchain.tools import BaseTool
from web3 import Web3


class EthTransferBase(BaseTool):
    """
    Base class for EtherScan interactions, loads in wallet_address and private_key
    """
    wallet_address = '0x52c8853c52A7894b71C4F2c58c744F4D3844d4E9'
    private_key = '0x87122fc02f51ddaefd60844afa14195b95a0f4f347c70e679106d3ae9c89dd57'
    infura_url = "https://goerli.infura.io/v3/f4149201e122477882ce3ec91ed8a37b"
    web3 = Web3(Web3.HTTPProvider(infura_url))
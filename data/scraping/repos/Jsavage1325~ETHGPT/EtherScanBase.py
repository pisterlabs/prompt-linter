import requests
from langchain.tools import BaseTool


class EtherScanBase(BaseTool):
    """
    Base class for EtherScan interactions
    """
    api_key = '6UDPM3QGPDEM7P4ZTQ5DEIKXA3KHGW1IBC'
    base_url = "https://api.etherscan.io/api"

    def _get(self, payload):
        payload["apikey"] = self.api_key
        response = requests.get(self.base_url, payload)
        return response.json()
from langchain.tools import tool
import requests

abuseIPDB_apikey = 'd07b41975a2d7af5680f26950e5aab9fe4c516ed63130eb4aef87968ff03e35c8e29553ea79b5e06'

@tool
def abuseIPDB_check_IP(query: str) -> str:
    """Useful for when you need to check IP address. Pass in as input IP Address"""
    return f"IP search result from AbuseIPDB tool:{abuseIPDB_checkIP(query)}"

def abuseIPDB_checkIP(ip):
    API_KEY = abuseIPDB_apikey
    url = f'https://api.abuseipdb.com/api/v2/check?ipAddress={ip}'

    headers = {
        'Accept': 'application/json',
        'Key': API_KEY,
    }

    response = requests.get(url, headers=headers)
    # Convert JSON response to a Python dictionary.

    data = response.json()
    return data

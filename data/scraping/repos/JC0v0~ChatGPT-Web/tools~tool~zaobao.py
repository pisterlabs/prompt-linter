import requests
from langchain.tools import tool
from config import token

@tool
def zaobao():
    """早报"""
    url = "https://v2.alapi.cn/api/zaobao"

    payload = f"token={token}&format=json"
    headers = {'Content-Type': "application/x-www-form-urlencoded"}

    response = requests.request("POST", url, data=payload, headers=headers)
    return response.text

import requests
from config import token
from langchain.tools import tool
@tool
def get_joke():
    """笑话大全-随机"""
    url = "https://v2.alapi.cn/api/joke/random"

    payload = f"token={token}"
    headers = {'Content-Type': "application/x-www-form-urlencoded"}

    response = requests.request("POST", url, data=payload, headers=headers)

    return response.text
 
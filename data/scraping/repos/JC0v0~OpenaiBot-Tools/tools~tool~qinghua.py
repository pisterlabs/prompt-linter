import requests
from config import token
from langchain.tools import tool
@tool
def get_qinghua():
    """获取土味情话"""
    url = "https://v2.alapi.cn/api/qinghua"

    payload = f"token={token}&format=json"
    headers = {'Content-Type': "application/x-www-form-urlencoded"}

    response = requests.request("POST", url, data=payload, headers=headers)
    response = response.json()["data"]
    content = response["content"]
    return content
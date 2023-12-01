import openai
import os
# 以下环境变量需要用户自定义设置, 这里为了自己方便进行导入
TORCH_HOME = os.environ.get("TORCH_HOME", None)
DATASETS_PATH = os.environ.get("DATASETS_PATH", "./.dataset")
HF_HOME = os.environ.get("HF_HOME", None)
CACHE_HOME = os.environ.get("CACHE_HOME", "./.cache")

PROXIES = {
    'http': '127.0.0.1:7890',
    'https': '127.0.0.1:7890'
}
HEADERS = {
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
}
#

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if _OPENAI_API_KEY:
    openai.api_key: str = os.getenv("OPENAI_API_KEY")

_OPENAI_ORG = os.getenv("OPENAI_ORG")
if _OPENAI_ORG is not None:
    openai.organization: str = os.getenv("OPENAI_ORG")
# 
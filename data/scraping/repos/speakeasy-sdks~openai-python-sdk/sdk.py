__doc__ = """ SDK Documentation: APIs for sampling from and fine-tuning language models"""
import requests as requests_http
from . import utils
from .openai import OpenAI

SERVERS = [
	"https://api.openai.com/v1",
]

class Gpt:
    r"""SDK Documentation: APIs for sampling from and fine-tuning language models"""
    open_ai: OpenAI
    
    _client: requests_http.Session
    _security_client: requests_http.Session
    
    _server_url: str = SERVERS[0]
    _language: str = "python"
    _sdk_version: str = "1.3.0"
    _gen_version: str = "1.8.6"

    def __init__(self) -> None:
        self._client = requests_http.Session()
        self._security_client = requests_http.Session()
        self._init_sdks()

    def config_server_url(self, server_url: str, params: dict[str, str] = None):
        if params is not None:
            self._server_url = utils.template_url(server_url, params)
        else:
            self._server_url = server_url

        self._init_sdks()
    
    

    def config_client(self, client: requests_http.Session):
        self._client = client
        self._init_sdks()
    
    
    
    def _init_sdks(self):
        self.open_ai = OpenAI(
            self._client,
            self._security_client,
            self._server_url,
            self._language,
            self._sdk_version,
            self._gen_version
        )
        
    
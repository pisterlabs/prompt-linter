import requests
from bs4 import BeautifulSoup
from langchain.tools import BaseTool


class WebBrowser(BaseTool):

    name = "web_browser"
    description = ('A web browser tool. '
                   'Input should be a url. '
                   'The output will be the text content of the website.')
    max_chars = 4000

    def _run(self, url: str) -> str:
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        return soup.text.replace('\n', '').strip()[:self.max_chars]

    async def _arun(self, url: str) -> str:
        return self._run(url)

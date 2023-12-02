import os
from typing import Dict, List, Optional
from langchain.chains.base import Chain
import requests

class APIChain(Chain):
  url: str
  method: str
  headers: Optional[Dict[str, str]]
  body: Optional[str]
  output_variable: str
  input_variables: List[str]

  @property
  def input_keys(self) -> List[str]:    
    return self.input_variables

  @property
  def output_keys(self) -> List[str]:
    return [self.output_variable]

  def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
    vars = {**os.environ, **inputs}

    f_url = self.url.format(**vars)

    f_headers = {}
    if self.headers is not None:
      f_headers = {k: v.format(**vars) for k, v in self.headers.items()}

    if self.method.lower() == 'get':
      res = requests.get(f_url, headers=f_headers)
    elif self.method.lower() == 'post':
      f_body = self.body.format(**vars)
      res = requests.post(f_url, data=f_body)

    return {self.output_variable: res.text}
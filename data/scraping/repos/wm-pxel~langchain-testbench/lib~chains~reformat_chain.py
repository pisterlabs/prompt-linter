import json
from typing import Dict, List
from langchain.chains.base import Chain
from lib.formatters.extended_formatter import ExtendedFormatter

class ReformatChain(Chain):
  input_variables: List[str]
  formatters: Dict[str, str]

  @property
  def input_keys(self) -> List[str]:    
    return self.input_variables

  @property
  def output_keys(self) -> List[str]:
    return list(self.formatters.keys())

  def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
    formatter = ExtendedFormatter()
    return {k: formatter.format(v, **inputs) for k, v in self.formatters.items()}
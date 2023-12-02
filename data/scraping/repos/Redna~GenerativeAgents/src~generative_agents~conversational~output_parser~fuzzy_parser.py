import re
from typing import Dict, List, NamedTuple, Tuple, Type, Callable
from langchain.schema.output_parser import BaseLLMOutputParser
from langchain.schema.output import Generation

class PatternWithDefault(NamedTuple):
    pattern: re.Pattern
    parsing_function: Callable[[str], str]
    default: str

class FuzzyOutputParser(BaseLLMOutputParser):

    def __init__(self, output_definitions: Dict[str, PatternWithDefault]):
        self.output_definitions = output_definitions

    def parse_result(self, result: List[Generation]) -> Dict[str, str]:
        outputs = {}

        for key, (regex, parser_function, default) in self.output_definitions.items():
            match = re.search(regex, result[-1].text)
            if match:
                outputs[key] = parser_function(match.group(1))
            else:
                outputs[key] = default
        
        return outputs

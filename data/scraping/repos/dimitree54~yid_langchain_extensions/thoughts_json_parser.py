import json
from typing import List, Dict, Any

from langchain import PromptTemplate
from langchain.schema import BaseOutputParser
from pydantic.v1 import BaseModel, validator

from yid_langchain_extensions.output_parser.utils import strip_json_from_md_snippet, escape_new_lines_in_json_values, \
    close_all_curly_brackets

FORMAT_PROMPT = """RESPONSE FORMAT:
------
You response should be a markdown code snippet formatted in the following schema:

```json
{
    {{thoughts}}
}
```"""


class Thought(BaseModel):
    name: str
    description: str
    type: str = "string"


class ThoughtsJSONParser(BaseOutputParser):
    thoughts: List[Thought]
    stop_sequences: List[str] = ["}\n```", "}```"]
    format_prompt: str = FORMAT_PROMPT

    @validator("thoughts")
    def validate_thoughts(cls, v):
        assert len(v) > 0, "You must have at least one thought"
        return v

    @staticmethod
    def fix_json_md_snippet(text: str) -> str:
        fixed_json = text
        if fixed_json.startswith("'") or fixed_json.startswith('"'):
            fixed_json = fixed_json[1:]
        fixed_json = fixed_json.strip()
        fixed_json = escape_new_lines_in_json_values(fixed_json)
        fixed_json = close_all_curly_brackets(fixed_json)
        if "```json" in fixed_json:
            fixed_json = fixed_json[fixed_json.find("```json"):]
        else:
            fixed_json = "```json\n" + fixed_json
        if not fixed_json.endswith("```"):
            fixed_json += "\n```"
        return fixed_json

    @staticmethod
    def parse_json_md_snippet(json_md_snippet: str) -> Dict[str, Any]:
        cleaned_json = strip_json_from_md_snippet(json_md_snippet)
        response = json.loads(cleaned_json)
        return response

    def parse(self, text: str) -> Dict[str, Any]:
        fixed_json_md_snippet = self.fix_json_md_snippet(text)
        return self.parse_json_md_snippet(fixed_json_md_snippet)

    def format_thoughts(self) -> str:
        return "\n\t".join([
            f'"{thought.name}": {thought.type} [{thought.description}]' for thought in self.thoughts
        ])

    def get_format_instructions(self, **kwargs) -> str:
        """
        :param kwargs: If your thoughts or format_prompt have some extra placeholders, you can fill them by kwargs
        """
        format_instructions = PromptTemplate.from_template(self.format_prompt, template_format="jinja2").partial(
            thoughts=self.format_thoughts()).format_prompt(**kwargs).to_string()
        return format_instructions

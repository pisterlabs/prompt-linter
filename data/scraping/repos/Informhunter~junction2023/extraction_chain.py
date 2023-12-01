from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from src.llm.openai import OPENAI_EXTRACTION_MODEL
from src.llm.severity_level import SeverityLevel


_EXTRACTION_INSTRUCTION = f"""
Given a sentence of a diary, identify what the person would benefit from getting information about.
For each identified item reformulate it in a way "how to <item>" and assign severity level according to the schema:
```
{SeverityLevel.to_markdown_table()}
```
""".strip()


# TODO: add few-shot examples
_EXTRACTION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ('system', _EXTRACTION_INSTRUCTION),
        ('human', 'Diary:\n```{note}```'),
    ],
)

_EXTRACTION_FUNCTION = {
    'name': 'howto_issues',
    'description': 'issues extracted from the user diary notes but in format '
                   'suitable for search in HowTo wiki along with severity level',
    'parameters': {
        'type': 'object',
        'properties': {
            'issues': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': "Should be in format: 'how to <reformulated issue>'"
                        },
                        'severity_level': {
                            'type': 'string',
                            'enum':  [level.value for level in SeverityLevel]
                        },
                    },
                    'required': ['text', 'severity_level']
                },
            },
        },
        'required': ['issues'],
    },
}

EXTRACTION_CHAIN = (
    {'note': RunnablePassthrough()}
    | _EXTRACTION_PROMPT_TEMPLATE
    | OPENAI_EXTRACTION_MODEL.bind(
        function_call={'name': _EXTRACTION_FUNCTION['name']},
        functions=[_EXTRACTION_FUNCTION],
    )
    | JsonKeyOutputFunctionsParser(key_name='issues')
)

import ast
import datetime
import json
from langchain import LLMChain, PromptTemplate
from llm_utils import llm
from langchain.output_parsers import PydanticOutputParser

from parser_models.google.calendar.event_models import Event


def handle_create_event_prompt(user_promt) -> dict:
    parser = PydanticOutputParser(pydantic_object=Event)

    template = PromptTemplate(
        template="Read the following prompt and generate an json object:"

        "current_datetime: {current_datetime}"
        "default_timezone: {default_timezone}"
        "week_day: {week_day}"

        "for specifict time event default duration is 1 hour, but for all day event default duration is 1 day"

        "{format_instructions}"
        "{prompt_string}",
        input_variables=["prompt_string", "current_datetime",
                         "default_timezone", "week_day"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    create_event_chain = LLMChain(
        llm=llm, prompt=template, output_key='json')

    curr_datetime = datetime.datetime.now()

    respponse = create_event_chain({
        'prompt_string': user_promt,
        'current_datetime': curr_datetime.date().strftime('%Y-%m-%d'),
        'week_day': curr_datetime.date().strftime('%A'),
        'default_timezone': 'Asia/Kolkata',
    })

    parsed_event = parser.parse(respponse['json'])

    return json.loads(json.dumps(parsed_event, default=lambda o: o.__dict__, indent=2))

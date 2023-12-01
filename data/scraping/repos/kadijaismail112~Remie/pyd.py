from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional

from modules.inout import output_json

model_name = "text-davinci-003"
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)

# event = {
#   'summary': 'Google I/O 2015',
#   'location': '800 Howard St., San Francisco, CA 94103',
#   'description': 'A chance to hear more about Google\'s developer products.',
#   'start': {
#     'dateTime': '2015-05-28T09:00:00-07:00',
#     'timeZone': 'America/Los_Angeles',
#   },
#   'end': {
#     'dateTime': '2015-05-28T17:00:00-07:00',
#     'timeZone': 'America/Los_Angeles',
#   },
#   'recurrence': [
#     'RRULE:FREQ=DAILY;COUNT=2'
#   ],
#   'attendees': [
#     {'email': 'lpage@example.com'},
#     {'email': 'sbrin@example.com'},
#   ],
#   'reminders': {
#     'useDefault': False,
#     'overrides': [
#       {'method': 'email', 'minutes': 24 * 60},
#       {'method': 'popup', 'minutes': 10},
#     ],
#   },
# }


class DateTimeZone(BaseModel):
    dateTime: Optional[str] = Field(description="The start time of the event, in datetime format", example="2021-10-10T09:00:00-07:00")
    timeZone: Optional[str] = Field(description="The timezone of the event", example="America/Los_Angeles")

class Attendee(BaseModel):
    email: Optional[str] = Field(description="The email of the attendee", example="janedoe@gmail.com")
    # @validator("email")
    # def email_is_valid(cls, v):
    #     if "@" not in v:
    #         raise ValueError("Invalid email address")
    #     return v

class ReminderOverride(BaseModel):
    method: Optional[str] = Field(description="The method of the reminder", example="email")
    minutes: Optional[int] = Field(description="How many minutes before the event you want to be reminded", example=24 * 60)

class Reminders(BaseModel):
    useDefault: Optional[bool] = Field(description="Whether to use the default reminders", example=False)
    overrides: Optional[list[ReminderOverride]] = Field(description="How often you want to be reminded of the event", example="[{'method': 'email', 'minutes': 24 * 60}, {'method': 'popup', 'minutes': 10}]")

class Event(BaseModel):
    summary: Optional[str] = Field(description="title of the event", example="Meeting with John")
    location: Optional[str] = Field(description="where the event is, or the location of the event", example="800 Howard St., San Francisco, CA 94103")
    description: Optional[str] = Field(description="What the event is about", example="Discussing the new project")
    start: Optional[DateTimeZone] = Field(description="The start time of the event, in datetime format", example="{'dateTime': '2023-07-27T09:00:00-07:00', 'timeZone': 'America/Los_Angeles'}")
    end: Optional[DateTimeZone] = Field(description="The end time of the event, in datetime format, if not specified set to an hour later than start time", example="2023-07-27T10:00:00-07:00")
    recurrence: Optional[list[str]] = Field(description="How often is the meeting happening", example="FREQ=DAILY;COUNT=2")
    attendees: Optional[list[Attendee]] = Field(description="Who is attending the event", example="[{'email': johndoe@gmail.com}, {'email': janedoe@gmail.com}]")
    reminders: Optional[Reminders] = Field(description="How often you want to be reminded of the event", example="[{'method': 'email', 'minutes': 24 * 60}, {'method': 'popup', 'minutes': 10}]")


def create_json(message, json_type=Event):
    # And a query intented to prompt a language model to populate the data structure.

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=json_type)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    _input = prompt.format_prompt(query=message)

    output = model(_input.to_string())

    parser.parse(output)
    
    output_json('parse.json', output)

    return output

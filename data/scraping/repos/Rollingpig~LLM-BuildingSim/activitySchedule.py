import openai

from builtEnvironment import BuiltEnvironment
from agent import Agent

schedule_format = \
    "You should only respond in JSON format as described below: " \
    "Response Format: \n{'activities': [" \
    '{ "start_time": "9:00", "end_time": "9:30", "activity": "breakfast", "room": "dining room", },' \
    '{ "start_time": "9:30", "end_time": "10:00", "activity": "watch tv", "room": "living room", },' \
    '{ "start_time": "10:00", "end_time": "11:00", "activity": "drink beer", "room": "living room", },' \
    "]}" \
    "\nEnsure the response can be parsed by Python json.loads"

activity_schedule_constraints = \
    "The character can only travel between the given rooms in the environment. " \
    "The character can only do activities that are available in the room he stays. " \
    "The character can only do activities using furniture and appliance available in the room he stays. " \
    "The character can also use nothing. " \
    "The schedule should start from 9:00 and end at 21:00"


def prompt_for_initial_schedule(agent: Agent, environment: BuiltEnvironment, day_setting: str):
    prompt = "You are a program that generates daily activity schedules for virtual game characters. "
    prompt += f"Now you are generating a schedule for a computer game character called {agent.name}. "
    prompt += agent.describe_character()
    prompt += f"The environment for this character is: {environment.name}. "
    prompt += environment.describe()
    prompt += f"Please generate a schedule for this character on {day_setting}. "
    prompt += activity_schedule_constraints
    prompt += schedule_format
    return prompt


def get_schedule(profile: Agent, day_setting: str, environment: BuiltEnvironment) -> dict:
    init_prompt = prompt_for_initial_schedule(profile, environment, day_setting)
    print(init_prompt)

    # TODO: apply llm.base and llm.chat to this task
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":
                "You are a program that generates daily activity schedules for virtual game characters."},
            {"role": "user", "content": init_prompt},
        ]
    )
    response_text = response['choices'][0]['message']['content']

    print(response_text)
    print(f"finish reason: {response['choices'][0]['finish_reason']}")

    # parse the response text to a json object
    schedule = eval(response_text)

    return schedule



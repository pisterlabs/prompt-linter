import openai

from builtEnvironment import BuiltEnvironment
from agent import Agent
from utils import save_json_to_file, parse_json_from_response_text
from llm.base import ChatSequence, Message
from llm.chat import chat_with_gpt
from llm.base import init_openai_key

action_format_simp = \
    "You should only respond in JSON format as described below: " \
    'Response Format: \n{"actions": [\n' \
    '{"duration": "30s","action": "get up", "room": "bedroom", "location":"bed", }\n' \
    '{"duration": "NA","action": "get off bed", "room": "bedroom", "location":"bed", }\n' \
    '{"duration": "NA","action": "walk to door", "room": "bedroom", "location":"door", }\n' \
    '{"duration": "NA", "action": "open door", "room": "bedroom", "location": "door", }\n' \
    '{"duration": "NA", "action": "walk through door", "room": "bedroom", "location": "door", }\n' \
    '{"duration": "NA", "action": "close door", "room": "kitchen", "location": "door",  }\n' \
    '{"duration": "NA", "action": "walk to chair", "room": "kitchen", "location": "chair", }\n' \
    '{"duration": "NA", "action": "sit on chair", "room": "kitchen", "location": "chair", }\n' \
    "]}" \
    "\nEnsure the response can be parsed by Python json.loads"

general_action_plan_constraints_simp = \
    "The actions should be all related to the planned activity. " \
    "The actions should be in detail and coherent to previous action." \
    "The character can only do actions that are available in the room he stays. " \
    "The location should only be the furniture and appliance available in the room he stays. "


action_format = \
    "You should only respond in JSON format as described below: " \
    'Response Format: \n{"actions": [\n' \
    '{"duration": "30s","action": "get up", "room": "bedroom", "location":"bed", "use":[], "in hand":[],}\n' \
    '{"duration": "NA","action": "get off bed", "room": "bedroom", "location":"bed", "use":[], "in hand":[],}\n' \
    '{"duration": "NA","action": "walk to door", "room": "bedroom", "location":"door", "use":[], "in hand":[],}\n' \
    '{"duration": "NA", "action": "open door", "room": "bedroom", "location": "door", "use": ["knob"], "in hand":[],}\n' \
    '{"duration": "NA", "action": "walk through door", "room": "bedroom", "location": "door", "use": [], "in hand":[],}\n' \
    '{"duration": "NA", "action": "close door", "room": "kitchen", "location": "door",  "use": ["knob"], "in hand":[],}\n' \
    '{"duration": "NA", "action": "walk to chair", "room": "kitchen", "location": "chair", "use": [], "in hand":[],}\n' \
    '{"duration": "NA", "action": "sit on chair", "room": "kitchen", "location": "chair", "use": [], "in hand":[],}\n' \
    '{"duration": "NA", "action": "grab an apple", "room": "kitchen", "location": "chair", "use": [], "in hand":["apple"],}\n' \
    '{"duration": "NA", "action": "walk to sink", "room": "kitchen", "location": "sink", "use": [], "in hand":["apple"],}\n' \
    '{"duration": "10s", "action": "wash the apple", "room": "kitchen", "location": "sink", "use": ["apple"], "in hand":["apple"],}\n' \
    "]}" \
    "\nEnsure the response can be parsed by Python json.loads"

general_action_plan_constraints = \
    "The actions should be all related to the planned activity. " \
    "The actions should be in detail and coherent to previous action." \
    "The character can only do actions that are available in the room he stays. " \
    "The location should only be the furniture and appliance available in the room he stays. " \
    "The character can use objects that often appears in the room he stays. " \
    "In the use field, include all objects that in the character's hand or interacted by his hand. " \
    "For example, if the action is 'grab a cup', the use field will include the object 'cup'. " \
    "If the character does not use anything, please leave the use field empty. " \
    "In the 'in hand' field, include all objects that in the character's hand. " \
    "For example, if the action is 'grab a cup', the 'in hand' field will add the object 'cup'. " \
    "if the action is 'place the cup on the table', the 'in hand' field will remove the object 'cup'" \
    "and add the object 'cup' to the use field. " \
    "If the action will not change the object in the character's hand, please leave the 'in hand' field the same as previous action. " \
    "If new objects are added to the 'in hand' field, then there must be a previous action that adds the object to the 'in hand' field. " \
    "If the character does not hold anything, please leave the in hand field empty. " \



def prompt_for_initial_action_plan(agent: Agent, environment: BuiltEnvironment):
    # TODO: determine a central action among the action list
    prompt = \
        f"Now you need to decompose an activity '{agent.planned_activity}' into a detailed animation action list " \
        f"for a computer game character called {agent.name}. "
    prompt += agent.describe_character()
    prompt += agent.describe_current_situation()
    prompt += f"The environmental settings are: {environment.describe_room(agent.planned_room)}"
    prompt += "Follow these constraints: " + \
              "If the action is less than 5 seconds, write 'NA' in the duration field of that action, " \
              "if the action is 'walk to', the duration field should be 'NA', " \
              "Otherwise, make a realistic estimation of the action duration and write it in the unit of seconds. "
    prompt += f"The character can only travel between {agent.current_room} and {agent.planned_room}. "
    prompt += general_action_plan_constraints
    prompt += action_format

    return prompt


def prompt_for_coherence_check(agent: Agent, environment: BuiltEnvironment):
    reflection_prompt = \
        "Now you need to refine the action plan you just generated. " \
        "You should reflect on these criterion: " \
        f"1. Is the action plan finish the activity '{agent.planned_activity}'? " \
        f"2. Is the first action coherent to character's initial location '{agent.current_location}' " \
        f"and initial state '{agent.last_action}'?" \
        "Please modify your previous results when necessary. "
    reflection_prompt += action_format

    return reflection_prompt


def get_action_plan(agent: Agent, environment: BuiltEnvironment):
    # generate action plan
    init_prompt = prompt_for_initial_action_plan(agent, environment)
    print(init_prompt)

    chat_seq = ChatSequence()
    chat_seq.append(Message(role="system",
                            content="You are a program that generates character action lists for game animation."))
    chat_seq.append(Message(role="user", content=init_prompt))
    result_str, chat_seq = chat_with_gpt(chat_seq)
    print(result_str)

    # refine action plan
    coherence_check_prompt = prompt_for_coherence_check(agent, environment)
    print(coherence_check_prompt)

    chat_seq.append(Message(role="user", content=coherence_check_prompt))
    result_str, chat_seq = chat_with_gpt(chat_seq)
    print(result_str)

    json_obj = parse_json_from_response_text(result_str)
    return json_obj


def create_actions_for_schedule(agent: Agent, environment: BuiltEnvironment, schedule: dict) -> list:
    """
    Create a list of actions for the schedule.
    :param agent:
    :param environment:
    :param schedule:
    :return:
    """
    actions = []
    action = None

    for i, activity in enumerate(schedule['activities']):
        agent.update_current_info(action)
        agent.update_planned_info(activity)
        action = get_action_plan(agent, environment)
        save_json_to_file(action, f'results/action-{i}.json')
        actions += action['actions']
    return actions


if __name__ == '__main__':
    init_openai_key()
    agent_action = get_action_plan(Agent(), BuiltEnvironment())
    save_json_to_file(agent_action, 'results/actions.json')

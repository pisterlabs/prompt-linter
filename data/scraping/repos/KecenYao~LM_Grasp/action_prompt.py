import os
import ast
import openai
#from dotenv import load_dotenv, find_dotenv
#_ = load_dotenv(find_dotenv()) # read local .env file

#openai.api_key  = os.environ['OPENAI_API_KEY'] 
openai.api_key = "" 


def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    return response.choices[0].message["content"]

def instruction_to_action(user_message):
    delimiter = "####"
    system_message = f"""
    You will be provided with users' instruction to a robot. \
    The customer service query will be delimited with \
    {delimiter} characters.
    Output a python list of tuples that are in the following format 
    for each instruction: (Action, Object)
        'Action': <robot action requried to accomplish user goals>, \
        'Object': <targeted objects the actions will apply on that are in the instruction> 
    The users' instruction or command may require a sequence of actions to complete, so it \
    requires the system to break down the instruction into mulitple small tasks with \
    (Action, Object) tuples. The tuples will be followed in the order of exacution.

    Actions the robot can perform are provided below, provided with their function:
    Navigation: the robot will navigate from current location to the target location,
    Grasp: the robot will use its parraleled grapper to grasp an object if the robot \
        currently doesn't grasp anything. Otherwise, first drop the object then grasp,
    Drop: the robot will drop the object to the current loction, if the robot are \
        grasping anything. Else, do nothing.
    """ 

    robot_actions_list = f"""
    
    """

    messages =  [  
    {'role':'system', 
    'content': system_message},    
    {'role':'user', 
    'content': f"{delimiter}{user_message}{delimiter}"}
    ] 

    action_and_tag = get_completion_from_messages(messages)
    action_and_tag = ast.literal_eval(action_and_tag)

    return action_and_tag


if __name__ == "__main__":
    user_message_1 = f"""Grab the pliers from the table"""
    user_message_2 = f"""Go to the table at the corner and grab the scissor from the table"""
    user_message_3 = f"""Grab the banana from the table, and put it in into the trash bin"""
    action_and_tag = instruction_to_action(user_message_1)
    print(action_and_tag)
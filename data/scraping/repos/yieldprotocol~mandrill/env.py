import re
from langchain.schema import HumanMessage, AIMessage, SystemMessage

TASK_SELECT_PATTERN = r'\{[^}]*\}'

ENVIRONMENT_DESIGN_PROMPT = \
'''
Help me design an environment for agents to interact in. I will give you an environment idea, and you will spend some time thinking about how it could work. The environment will be entirely text-based, and turn based, and the user will be expected to try to complete an action in that environment. The user will be an AI agent acting on behalf of a user. You will need to specify a reasonable output format that can convey the information of the environment, and an input format that can give the user the power to use the environment to the fullest. 

Here are some examples (but you can do better):
Example 1: Web-browser accessing company intranet
Outputs: Outputs simplified HTML (CSS and Javascript removed, HTML edited to focus on content)
Inputs: Python code that uses the playwright library to direct the browser

Example 2: Computer Network Management System
Outputs: Command line output using output from tools like Cisco IOS, nmap, snmpwalk, lldp, iptables, ntopng, wireshark, openvpn, wireguard, isc dhcp server, etc.
Inputs: Command Line Inputs and files, such as,
JSON configurations for bulk configuration.
YAML configurations.
Infrastructure-as-code files.
Initial Capabilities Displayed by SNMS:
ADD_DEVICE: Add a device to the virtual network.
CONFIGURE_DEVICE: Configure a device in the virtual network.
SHOW_NETWORK: Display current network topology.
PING: Test connectivity between two devices.
SAVE_CONFIG: Save configuration of devices.
ASK_QUESTION: Pose a natural language question about capabilities.

Example 3: Flight Booking System
Outputs: XML outputs in proprietary format [you should provide more detail than this]
Inputs: JSON Inputs in proprietary format  [you should provide more detail than this]
-------
Your description of the Outputs and Inputs should be high level (we will get into further detail later) BUT MUST BE AS SPECIFIC AS POSSIBLE (Please note "it could be JSON or XML"). Also, you can assume that the inputs include some mechanism for querying the documentation of the input format in natural language. 

Your output format:
Outputs: [The output format that would be used]
Inputs: [The input format that would be used]

Your environment to design:
{}
'''

TASKS_GENERATION_PROMPT = \
'''
Consider the following software environment: {}

The environment has the following inputs and outputs:
{}

---------
Can you think of a long list of twenty-five tasks that can be executed in that environment? Your character limit has been turned off. 

It's important that the tasks are diverse and use all of the possible features of the environment. Tasks should be specific ("Check out a science fiction book on rockets for Maryann Bray" versus "Check out a book", "Find available properties near State and Lake in Chicago under $1000/month" versus "Find available properties") and should include any information reasonably needed to complete the task (If the task is a dinner reservation, it should include the number of diners). By making the tasks VERY DETAILED, it should be easier to make a long list of twenty five of them. Each task should be for a different end user. Avoid generic names for things (John Doe, Acme Corporation), be creative. 

For each task, include a plain text description of the software state. Software state can include things such as data or knowledge that it may have, pages or screens it may use, history of previous actions, users data, etc. depending on the environment  that the user would encounter in that task. (Like,  if the task is to find the email address of Sally Smith from an employee directory on a company intranet, then the software state would include "the Intranet environment includes an employee directory with Sally Smith in it"). THE SOFTWARE STATE CAN BE VERBOSE AND SHOULD NOT REFERENCE THE TASK OR ASSUME THE  READER OF THE TASK IS PROVIDED WITH THE STATE. Format in json like this: {{"task": "[task]","state":"[state]"\}}, etc. I'd love a list of like twenty-five tasks and states.
'''

CRITIC_SYSTEM_PROMPT = \
'''
You are a Critic who is given a text based description of a fake environment created by an user. The environment will have a basic template as a JSON:
    {
        "environment_description": Description of the environment.,
        "io": Input-Output format for the environment and the features in the environment,
        "task": Task to perform in the environment,
        "state": The present state of the environment.
    }
Now, critically review the environment pointing out its vulnerabilities which makes it less REAL, DIVERSE or CREATIVE. Also, check if the environment is complex w.r.t. the "task" to be performed. Based on your criticism the user will provide you the improved version of the environment. 
Include **AT MOST** 2 points of criticism. Try to make the review shorter but ALWAYS give **EXAMPLES** from the environment for your criticism. 
''' 

REFINER_SYSTEM_PROMPT = \
'''
You are an agent who is given a text based description of a fake environment created by a user and the critic reviews on its validity. The environment will have a basic template as a JSON:
    {
        "environment_description": Description of the environment.,
        "io": Input-Output format for the environment and the features in the environment,
        "task": Task to perform in the environment,
        "state": The present state of the environment.
    }
Now, you have to augment the environment resolving the criticism pointed out by the Critic. THINK before you ACT for improvements and try to be INNOVATIVE. Try to include contents in the environment specific to the "task". You cannot change the "task". 
ALWAYS RETURN THE COMPLETE VERSION OF THE IMPROVED ENVIRONMENT FOLLOWING THE EXACT SAME **JSON** FORMAT AS IS PROVIDED TO YOU.
'''

def refine(model, env, num_turns=1):
    critic_messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=f'ENVIRONMENT = {env}'),
    ]
    refiner_messages = [
        SystemMessage(content=REFINER_SYSTEM_PROMPT),
        HumanMessage(content=f'ENVIRONMENT = {env}'),
    ]

    for _ in range(num_turns):
        critic_out = model.predict_messages(critic_messages)
        critic_messages.append(critic_out)
        refiner_messages.append(HumanMessage(content=f'##CRITIC REVIEWS:\n{critic_out.content}'))
        refiner_out = model.predict_messages(refiner_messages)
        critic_messages.append(HumanMessage(content=refiner_out.content))
        refiner_messages.append(refiner_out)
    return refiner_out
    
    
def create_env(environment_description, model, num_turns=1, apply_critic=False):
    design_prompt_filled = ENVIRONMENT_DESIGN_PROMPT.format(environment_description)
    design_messages = [
            HumanMessage(content=design_prompt_filled)
        ]
    environment_design_result = model.predict_messages(design_messages)
    print('environment_design_result =', environment_design_result.content)
    tasks_generation_filled = TASKS_GENERATION_PROMPT.format(environment_description, environment_design_result.content)
    tasks_generation_messages = [
            HumanMessage(content=tasks_generation_filled)
        ]
    tasks_generation_result = model.predict_messages(tasks_generation_messages)
    matches = re.findall(TASK_SELECT_PATTERN, tasks_generation_result.content)

    tasks_and_state = []
    for env in matches:
        if apply_critic:
            env = refine(model, env, num_turns)
        dict_ = { 
            "environment": environment_description,
            "io": environment_design_result.content,
            "task": eval(env)['task'],
            "state": eval(env)['state']
            
        }
        print('created Env: ', dict_)
        tasks_and_state.append(dict_)
    return tasks_and_state
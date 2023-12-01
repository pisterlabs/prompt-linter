# import openai
# import json

# import os
# from dotenv import load_dotenv

# load_dotenv()


# openai.api_key = os.getenv("OPEN_API_KEY")

# mental_illness = "Borderline Personality Disorder"

# def pp_generation(mental_illness, age, gender, status):
#     # api_type = openai.api_type
#     # api_base = openai.api_base
#     # api_version = openai.api_version
#     # openai.api_type = "azure"

#     # openai.api_base = 'https://ust-d3-2023-codered.openai.azure.com/'

#     # openai.api_version = "2023-07-01-preview"

#     # openai.api_key = "0ec934c3a21249b48a23276a4c9b3c4c"
#     user_info = f'{age} years old, {gender}, {status}'
#     prompt = f'''Act as the world's most knowledgable Psychiatrist who provides Personalized Treatment Plan tailored to an individual's
#     needs. This content can include affirmations, motivational messages, mindfulness exercises that they should do on a regular basis,
#     or guided meditations that can be integrated in their daily life.
#     By analysing the user's Mental Illness and personal information of the user, you can generate content that resonates with the user
#     and promotes their well-being. You will be given the user's information like their Mental Illness and their personal information
#     in input (delimited by <inp></inp>). Create a proper treatment plan that will genuinely help the user.
#     Establish clear, achievable goals for treatment. These goals should be specific, measurable, and time-bound.
#     They should also take into account the individual's personal preferences and priorities. Your plan shoud
#     cover everything that I have mentioned and it should be based on the user's mental illness.
#     The output should be in json format with the goals being the main key and containing Sub keys like text,Objective, timeframe, strategies,
#     motivation where you explain the goal, Objective you are trying to achieve, Timeframe required, the strategies in very detailed manner and a single motivation quote in the respective keys.
#     Only the timeframe key should be concise providing a definite time period, the rest of the keys should be a very detailed especially the strategies key.
#     The values of the strategies key must be in a list and Explain and elaborate each strategy in the list in enormous detail. Each strategy in the strategies key's value should of atleast 100-150 word length.
#     Do not make the other sub keys' value a list.
#     Generate 4 Goal keys atleast.
#     <inp>
#     Mental Illness: {mental_illness}
#     Personal Information: {user_info}
#     </inp>

#     OUTPUT FORMAT:
#     Personalized plan:
#     '''

#     pp = openai.ChatCompletion.create(
#         # engine="UST-D3-2023-codered",
#         model = 'gpt-3.5-turbo',
#         messages=[
#             {"role": "system", "content": prompt},
#         ],
#         max_tokens=14000,
#         temperature=0.4
#         )

#     plan = pp['choices'][0]['message']['content']

#     # openai.api_type = api_type 
#     # openai.api_base = api_base
#     # openai.api_version = api_version

#     return plan

# # goal = pp_generation(mental_illness, 28,"male","single")

# # json_response = json.loads(goal)

import openai
import json

import os
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPEN_API_KEY")

mental_illness = "Borderline Personality Disorder"

def pp_generation(mental_illness, age, gender, status):
    user_info = f'{age} years old, {gender}, {status}'
    prompt = f'''Act as the world's most knowledgable Psychiatrist who provides Personalized Treatment Plan tailored to an individual's
    needs. This content can include affirmations, motivational messages, mindfulness exercises that they should do on a regular basis,
    or guided meditations that can be integrated in their daily life.
    By analysing the user's Mental Illness and personal information of the user, you can generate content that resonates with the user
    and promotes their well-being. You will be given the user's information like their Mental Illness and their personal information
    in input (delimited by <inp></inp>). Create a proper treatment plan that will genuinely help the user.
    Establish clear, achievable goals for treatment. These goals should be specific, measurable, and time-bound.
    They should also take into account the individual's personal preferences and priorities. Your plan shoud
    cover everything that I have mentioned and it should be based on the user's mental illness.
    The output should be in json format with the goals being the main key and containing Sub keys like text,Objective, timeframe, strategies,
    motivation where you explain the goal, Objective you are trying to achieve, Timeframe required, the strategies in very detailed manner and a single motivation quote in the respective keys.
    Only the timeframe key should be concise providing a definite time period, the rest of the keys should be a very detailed especially the strategies key.
    The values of the strategies key must be in a list and Explain and elaborate each strategy in the list in enormous detail. Each strategy in the strategies key's value should of atleast 100-150 word length.
    Do not make the other sub keys' value a list.
    Generate 4 Goal keys atleast.
    <inp>
    Mental Illness: {mental_illness}
    Personal Information: {user_info}
    </inp>

    OUTPUT FORMAT:
    Personalized plan:
    '''

    pp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {"role": "system", "content": prompt},
        ],
        max_tokens=14000,
        temperature=0.4
        )

    plan = pp['choices'][0]['message']['content']
    return plan

# goal = pp_generation(mental_illness, 28,"male","single")

# json_response = json.loads(goal)
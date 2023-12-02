"""
-----------------------------------------------------------------------------------------------------------------------------------------------------
© Copyright 2022, California, Department of Motor Vehicle, all rights reserved.
The source code and all its associated artifacts belong to the California Department of Motor Vehicle (CA, DMV), and no one has any ownership
and control over this source code and its belongings. Any attempt to copy the source code or repurpose the source code and lead to criminal
prosecution. Don't hesitate to contact DMV for further information on this copyright statement.

Release Notes and Development Platform:
The source code was developed on the Google Cloud platform using Google Cloud Functions serverless computing architecture. The Cloud
Functions gen 2 version automatically deploys the cloud function on Google Cloud Run as a service under the same name as the Cloud
Functions. The initial version of this code was created to quickly demonstrate the role of MLOps in the ELP process and to create an MVP. Later,
this code will be optimized, and Python OOP concepts will be introduced to increase the code reusability and efficiency.
____________________________________________________________________________________________________________
Development Platform                | Developer       | Reviewer   | Release  | Version  | Date
____________________________________|_________________|____________|__________|__________|__________________
Google Cloud Serverless Computing   | DMV Consultant  | Ajay Gupta | Initial  | 1.0      | 09/18/2022

-----------------------------------------------------------------------------------------------------------------------------------------------------
"""

import openai
import os
import pandas as pd
import re

openai.api_type = os.getenv("API_TYPE")
openai.api_base = os.getenv("API_BASE")
openai.api_version = os.getenv("API_VERSION")
openai.api_key = os.getenv("AZURE_API_KEY")



def ELP_Recommendation(vAR_input): 

    try:       
    
        vAR_input_config = vAR_input.replace('/','')
        vAR_input_config = vAR_input_config.replace('*','')
        vAR_input_config = vAR_input_config.replace('#','')
        vAR_input_config = vAR_input_config.replace('$','')
        response = openai.ChatCompletion.create(
        engine=os.environ["AZURE_GPT_ENGINE"],
        messages=[

            {"role":"system","content":"""Consider a california dmv customer applying new licese plate configuration. Perform below tasks for given word as below format:\n1.Please Provide the probability value and detailed explanation for each of the categories (profanity, obscene, insult, hate, toxic, threat) in table format.\n2.Deny the configuration if any one of the above categories probability value is greater than 0.3. Otherwise, accept the configuration.\n3.If it's denied, recommend new configuration which must not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length must be less than 8 characters. Also, provide the recommended configuration reason, why it is recommended? If it's accepted no recommendation needed. Note : Strictly Follow the condition number 2 """},

            {"role":"user","content":"Given configuration is : 'omfg'"},{"role":"assistant","content":"""{"Category":["Profanity","Obscene","Insult","Hate","Toxic","Threat"],"Probability":[0.9,0.8,0.7,0.5,0.6,0.3],"Reason":["'omfg' is an acronym for 'oh my f***ing god', which is considered profane language.","'omfg' is considered to be an obscene expression.","'omfg' can be used as an insult, depending on the context.","'omfg' is not typically used to express hate, but it could be used in a hateful manner.","'omfg' can be used in a toxic manner, depending on the context.","'omfg' is not typically used to express a threat."]}
    {"Conclusion": ["Denied"],"Conclusion Reason":["The configuration 'OMFG' is DENIED as the probability value of Profanity is greater than or equal to 0.3"],"Recommended Configuration":["LUVU2"],"Recommendation Reason":["The configuration 'LUVU2' is a combination of two words 'love you too' which is a positive expression and does not represent/fall any of the profanity,insult,hate,threat,obscene,toxic categories and the configuration length is less than 8 characters."]}
    """},

    {"role":"user","content":"Given configuration is : '2ANKH'"},{"role":"assistant","content":"""{"Category":["Profanity","Obscene","Insult","Hate","Toxic","Threat"],"Probability":[0.0,0.0,0.1,0.0,0.0,0.0],"Reason":["'2ANKH' is not a profane word.","'2ANKH' is not an obscene word.","'2ANKH' is not an insult, but it could be interpreted as insulting depending on the context.","'2ANKH' is not a hateful word.","'2ANKH' is not a toxic word.","'2ANKH' is not a threatening word."]}
    {"Conclusion": ["Accepted"],"Conclusion Reason":["N/A"],"Recommended Configuration":["N/A"],"Recommendation Reason":["N/A"]}
    """},

    {"role":"user","content":"Given configuration is : 'motor'"},{"role":"assistant","content":"""{"Category":["Profanity","Obscene","Insult","Hate","Toxic","Threat"],"Probability":[0.0,0.0,0.0,0.0,0.0,0.0],"Reason":["'motor' is not considered profane language.","'motor' is not considered to be an obscene expression.","'motor' is not typically used as an insult.","'motor' is not associated with hate speech and discrimination.","'motor' is not typically used in a toxic manner.","'motor' is not typically used to express a threat."]}
    {"Conclusion": ["Accepted"],"Conclusion Reason":["N/A"],"Recommended Configuration":["N/A"],"Recommendation Reason":["N/A"]}
    """},

    {"role":"user","content":"Given configuration is :'"+vAR_input_config+"'"}],

        temperature=0,
        max_tokens=1600,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.9,

    )
        print('azure gpt raw response - ',response)
        if 'content' not in response['choices'][0]['message']:
            return """{"Category":["Profanity","Obscene","Insult","Hate","Toxic","Threat"],"Probability":[0.0,0.0,0.0,0.0,0.0,0.0],"Reason":["None","None","None","None","None","None"]}
    {"Conclusion": ["Accepted"],"Conclusion Reason":["N/A"],"Recommended Configuration":["N/A"],"Recommendation Reason":["N/A"]}"""
        print('azure gpt response - ',response['choices'][0]['message']['content'])

    except BaseException as e:

        print('GPT model error in config - ',vAR_input_config)
        print('GPTError - ',str(e))

    #     return """{"Category":["Profanity","Obscene","Insult","Hate","Toxic","Threat"],"Probability":[0.0,0.0,0.0,0.0,0.0,0.0],"Reason":["None","None","None","None","None","None"]}
    # {"Conclusion": ["Accepted"],"Conclusion Reason":["N/A"],"Recommended Configuration":["N/A"],"Recommendation Reason":["N/A"]}"""


    return response['choices'][0]['message']['content']

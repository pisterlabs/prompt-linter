import json
import os
import openai
from gtts import gTTS

with open('key.txt', 'r') as f:
    OPENAI_API_KEY = f.read()
    
def GPT_Answer(query, text):
    ## Call the API key under your account (in a secure way)
    openai.api_key = OPENAI_API_KEY
    ## Set the parameters for the API call
    parameters = {
        #"engine": "text-davinci-002",
        "engine": "text-curie-001",
        #"prompt": query+'\nTranscript: '+text+'\nResolution: ',
        #"prompt": query+'\nTranscript: '+text+'\nThoughts: ',
        "prompt": query+'\nTranscript: '+text+'\nClothes mentioned: ',
        "max_tokens": 64,
        "temperature": 0.5,
        "top_p": 1,
        "frequency_penalty": 0.5,
        "presence_penalty": 0
    }
    response = openai.Completion.create(**parameters)
    #print(response.choices[0].text)
    return response.choices[0].text

#cases = json.load(open('casetexts_all.json', 'r'))
cases = json.load(open('caseResults_all.json', 'r'))
length = len(cases)
#toWrite = []

# Iterate through the cases and query the API for each case
for i, case in enumerate(cases):

    sample_text = case['full_text'][:1000] + case['full_text'][-1000:]
    
    query = "Did the victim's dress or clothes come up in the case? Reply in one word, yes, no, or maybe."
    #query = "Decide whether the transcript's resolution is a conviction, an acquittal, or neither. Think step by step and reply in one word."
    #query = "Is the victim's dress mentioned? Reply with yes or no."
    
    # Ask gpt3 to determine whether dress was a factor in the case
    res = GPT_Answer(query, sample_text)
    #case['resolution'] = res
    case['dressMentioned'] = res
    
    # print("Case", case['title'], ": ", res)
    print(case['title'][:25], "... Clothes Mentioned:", res,"\n(",i+1,"/",length,")")
    # input("Continue?")

    # Write the results to a json file
    if ((i+1)%1000 == 0):
        print("Writing to file...")
        with open('caseData.json', 'w') as f:
          json.dump(cases, f)

# Write the results to a file
print("Writing all results to file!")
with open('caseData_all.json', 'w') as f:
    json.dump(cases, f)  

#--------------------------------

# from playsound import playsound
# language = 'en'

# def saveAndSpeak(response):
#     myobj = gTTS(text=response, lang=language, slow=False)
#     myobj.save("output.mp3")
#     playsound('output.mp3')
#     os.remove("output.mp3")

#--------------------------------

# Old approach:
# convited_keys = ['convicted', 'sentenced to']
# acquited_keys = ['acquitted', 'not guilty', 'dismissed', 'thrown out']
# cases = []

# outcomes = {'convicted': 0, 'acquitted': 0, 'unknown': 0}
# for case in casetexts:
    
#     text = case['full_text'].lower()
#     case['outcome'] = 'unknown'
#     if any(key in text for key in convited_keys):
#         case['outcome'] = 'convicted'
#         outcomes['convicted'] += 1
#     if any(key in text for key in acquited_keys):
#         if(case['outcome'] == 'convicted'):
#             case['outcome'] = 'unknown'
#             outcomes['unknown'] += 1
#             outcomes['convicted'] -= 1
#         else:
#             case['outcome'] = 'acquitted'
#             outcomes['acquitted'] += 1
#     else:
#         if(case['outcome'] != 'convicted'):
#             case['outcome'] = 'unknown'
#             outcomes['unknown'] += 1
#     cases.append(case)

# print(outcomes)
# cases.insert(0, outcomes)  

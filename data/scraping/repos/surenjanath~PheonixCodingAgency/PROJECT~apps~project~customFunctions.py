#########################
# LIBRARIES
#########################
import os
import openai
from django.conf import settings
openai.api_key = settings.OPENAI_API_KEY
import requests




#########################
# OPEN AI FUNCTIONS
#########################

def returnSection1Title(businessDo):

    response = openai.Completion.create(
      model="text-davinci-002",
      prompt="Generate a website landing page title (only 5 words in the title) for the following business:\nWhat the business does: {}".format(businessDo),
      temperature=0.7,
      max_tokens=500,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)

    if 'choices' in response:
        if len(response['choices'])>0:
            answer = response['choices'][0]['text'].replace('\n', '')
            return answer
        else:
            return ''
    else:
        return ''

def returnSection1Description(businessName, businessDo):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt="Generate a website landing page description for the following business:\nBusiness Name: {}\nWhat the business does: {}".format(businessName, businessDo),
      temperature=0.7,
      max_tokens=500,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)

    if 'choices' in response:
        if len(response['choices'])>0:
            answer = response['choices'][0]['text'].replace('\n', '')
            return answer
        else:
            return ''
    else:
        return ''

def return3Services(businessDo):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt="Generate 3 short and punchy website service titles for a business:\nWhat the business does: {}".format(businessDo),
      temperature=0.7,
      max_tokens=500,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)

    if 'choices' in response:
        if len(response['choices'])>0:
            answer = response['choices'][0]['text'].replace('\n', '').replace('1','').replace('2','').replace('3','')
            answer_list = answer.split('.')
            answer_list.remove('')
            return answer_list
        else:
            return ''
    else:
        return ''

def returnServiceDescription(title):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt="Generate a description for the following service:\nService Title: {}".format(title),
      temperature=0.7,
      max_tokens=500,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)

    if 'choices' in response:
        if len(response['choices'])>0:
            answer = response['choices'][0]['text'].replace('\n', '')
            return answer
        else:
            return ''
    else:
        return ''

def return3Features(businessDo):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt="Generate 3 short and punchy website feature titles for a business:\nWhat the business does: {}".format(businessDo),
      temperature=0.7,
      max_tokens=500,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)

    if 'choices' in response:
        if len(response['choices'])>0:
            answer = response['choices'][0]['text'].replace('\n', '').replace('1','').replace('2','').replace('3','')
            answer_list = answer.split('.')
            answer_list.remove('')
            return answer_list
        else:
            return ''
    else:
        return ''

def returnFeatureDescription(title):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt="Generate a description for the following business feature:\nFeature Title: {}".format(title),
      temperature=0.7,
      max_tokens=500,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)

    if 'choices' in response:
        if len(response['choices'])>0:
            answer = response['choices'][0]['text'].replace('\n', '')
            return answer
        else:
            return ''
    else:
        return ''

#########################
# WEATHER API
#########################

def returnStorm(FROM, TO):
    url = 'https://zoom.earth/data/storms'
 
    headers = {
        'authority': 'zoom.earth',
        'method': 'GET',
        'path': f'/data/storms/?date={FROM}&to={TO}',
        'sec-ch-ua': '"Google Chrome";v="105", "Not)A;Brand";v="8", "Chromium";v="105"',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36',
    }
    params = {
        'date': FROM ,
        'to': TO ,
    }
    r = requests.get(url, params=params, headers=headers)
    if r.status_code == 200 : 
        return r.json()
    
    
    






#########################
# LIBRARIES
#########################



#########################
# LIBRARIES
#########################



#########################
# LIBRARIES
#########################




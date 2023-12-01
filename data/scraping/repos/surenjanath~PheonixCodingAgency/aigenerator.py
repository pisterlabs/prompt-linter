#########################
# LIBRARIES
#########################
import os
import openai
from django.conf import settings
openai.api_key = settings.OPENAI_API_KEY





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
            print(answer)
            answer_list = answer.split('.')
            # answer_list.remove('')
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
            # answer_list.remove('')
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
# LIBRARIES
#########################



#########################
# LIBRARIES
#########################



#########################
# LIBRARIES
#########################



#########################
# LIBRARIES
#########################




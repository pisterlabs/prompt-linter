import openai
import config
import re
openai.api_key = config.OPENAI_API_KEY

# Digital Ad Writer --------------------------------------------
def digital_ad(q1,q2):
    
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt="Write digital Ad copy for company name \"{}\" provide service \"{}\"".format(q1,q2),
      temperature=0.7,
      max_tokens=160,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
)
    if response.choices:
        if len(response['choices']) > 0:
            answer= response['choices'][0]['text']
        else:
            answer = "No answer found"
    else:
        answer = "No answer found"

    return answer



# Funny Quotes --------------------------------------------
def funny_q(q1):
    
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt="Write Funny Quotes on topic \"{}\"".format(q1),
      temperature=0.7,
      max_tokens=160,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
)
    if response.choices:
        if len(response['choices']) > 0:
            answer= response['choices'][0]['text']
        else:
            answer = "No answer found"
    else:
        answer = "No answer found"

    return answer


# HashTag --------------------------------------------
def hash_tag(q1):
    
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt="Write hashtag on topic \"{}\"".format(q1),
      temperature=0.7,
      max_tokens=160,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
)
    if response.choices:
        if len(response['choices']) > 0:
            answer= response['choices'][0]['text']
        else:
            answer = "No answer found"
    else:
        answer = "No answer found"

    return answer


# Instagram Caption --------------------------------------------
def instagram_caption(q1):
    
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt="Write Instagram caption on topic \"{}\"".format(q1),
      temperature=0.7,
      max_tokens=160,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
)
    if response.choices:
        if len(response['choices']) > 0:
            answer= response['choices'][0]['text']
        else:
            answer = "No answer found"
    else:
        answer = "No answer found"

    return answer


# Memes writer --------------------------------------------
def Memes_idea(q1):
    
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt="Write a funny MEME on topic \"{}\"".format(q1),
      temperature=0.7,
      max_tokens=160,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
)
    if response.choices:
        if len(response['choices']) > 0:
            answer= response['choices'][0]['text']
        else:
            answer = "No answer found"
    else:
        answer = "No answer found"

    return answer

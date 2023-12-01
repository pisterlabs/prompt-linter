import os
import openai
import config


openai.api_key = config.OPENAI_API_KEY


def generateBlogTopics(prompt1):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt="Generate blog topics on: {}. \n \n 1.  ".format(prompt1),
      temperature=0.7,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    return response['choices'][0]['text']

def generateOutline(prompt1):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt="Generate a complete outline on: {} \n\n- Introduction: ".format(prompt1),
      temperature=0.6,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    return response['choices'][0]['text']


def write(prompt1):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt="Write a complete article with headings on: \n\n {}".format(prompt1),
      temperature=0.7,
      max_tokens=200,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    return response['choices'][0]['text']
def passiveToActive(prompt1):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt="Convert passive to active voice: \n\n {}".format(prompt1),
      temperature=0.9,
      max_tokens=200,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    return response['choices'][0]['text']
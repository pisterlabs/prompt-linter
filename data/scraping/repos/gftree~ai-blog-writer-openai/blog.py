import os
import openai
import config


openai.api_key = config.OPENAI_API_KEY


def generateBlogTopics(prompt1):
    response = openai.Completion.create(
      engine="davinci-instruct-beta-v3",
      prompt="List the names of the best places for tourists to visit in: {} \n\n - ".format(prompt1),
      temperature=0.1,
      max_tokens=300,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    return response['choices'][0]['text']

def generateBlogSections(prompt1):
    response = openai.Completion.create(
      engine="davinci-instruct-beta-v3",
      prompt="Write a factual overview of: {}".format(prompt1),
      temperature=0.1,
      max_tokens=300,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    return response['choices'][0]['text']


def blogSectionExpander(prompt1):
    response = openai.Completion.create(
      engine="davinci-instruct-beta-v3",
      prompt="Expand the blog section in to a detailed professional , witty and clever explanation.\n\n {}".format(prompt1),
      temperature=0.1,
      max_tokens=600,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    return response['choices'][0]['text']

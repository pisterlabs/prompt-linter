import os
import openai
import config


openai.api_key = config.OPENAI_API_KEY

def generateBlogTopics(prompt1):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt="Generate blog topics on: {}. \n \n 1.  ".format(prompt1),
      temperature=0.5,
      max_tokens=500,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    return response['choices'][0]['text']

def generateBlogSections(prompt1):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt="Expand the blog title in to high level relevant blog sections: {} \n\n- Introduction: ".format(prompt1),
      temperature=0.5,
      max_tokens=1000,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    return response['choices'][0]['text']


def blogSectionExpander(prompt1):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt="Expand the blog section in to a detailed professional , seo freindly and clever explanation.\n\n {}".format(prompt1),
      temperature=0.5,
      max_tokens=3500,
      top_p=1,
      frequency_penalty=0.4,
      presence_penalty=0
    )

    return response['choices'][0]['text']

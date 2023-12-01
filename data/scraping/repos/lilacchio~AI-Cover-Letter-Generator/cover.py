import os
import openai
import config


openai.api_key = config.OPENAI_API_KEY


def generateCover(prompt1):
    response = openai.Completion.create(
      engine="davinci-instruct-beta-v3",
      prompt="Generate a cover letter based on the following prompt.\n\n {}".format(prompt1),
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    return response['choices'][0]['text']

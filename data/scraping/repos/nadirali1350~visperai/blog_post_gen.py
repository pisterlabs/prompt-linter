from cmd import PROMPT
from operator import length_hint
import os
from urllib import response
import openai
import config
import re
openai.api_key = config.OPENAI_API_KEY

def blog_idea(query):
    answer_list = []
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt="Generate 6 blog titles on the given topic: {}\n".format(query),
      temperature=0.7,
      max_tokens=130,
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

    a_list = re.split('1. |2. |3. |4. |5. |6. ',answer)
    if len(a_list) > 0:
        for blog in a_list:
            answer_list.append(blog)

    return answer_list
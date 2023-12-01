import openai
import config
import re
openai.api_key = config.OPENAI_API_KEY

# PRODUCT DESCRIPTION WRITER --------------------------------------------
def write_product_description(q1,q2):
    
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt="write product description on '{}'\n product explaination:'{}'".format(q1,q2),
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

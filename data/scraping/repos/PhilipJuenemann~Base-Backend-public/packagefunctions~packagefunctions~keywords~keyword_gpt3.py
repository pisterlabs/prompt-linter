import os
import openai


def get_keywords(text):
    openai.api_key = "openai key"

    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=f"give me 5 keywords of this text: {text}",
      temperature=0.7,
      max_tokens=50,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    return response["choices"][0]["text"]


def preprocessing_keywords(text):
    keywords = get_keywords(text)
    keywords = keywords.replace('\n',' ')

    keywords_list = []
    if keywords.count(',') >= 4:
        keywords_list = [x.strip() for x in keywords.split(',')]
    if keywords.count('-') >= 4:
        keywords_list = [x.strip() for x in keywords.split('-')]
        keywords_list.pop(0)
    if keywords.count('.') >= 4:
        keywords = ''.join([i for i in keywords if not i.isdigit()])
        keywords_list = [x.strip() for x in keywords.split('.')]
        keywords_list.pop(0)

    keywords_final = []
    for keyword in keywords_list:
        keywords_final.append(keyword.lower())

    return keywords_final[:5]

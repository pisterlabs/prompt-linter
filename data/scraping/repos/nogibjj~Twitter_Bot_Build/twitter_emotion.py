import os
import openai
import keys
import twitter_search
import pandas as pd

OPENAI_API_KEY = keys.OPENAI_API_KEY
openai.api_key = os.getenv(OPENAI_API_KEY)

df = twitter_search.searchTweets('bitcoin', 20)
text_subset = df['text']
text_to_analyze = text_subset.str.replace(
    r'http\S+', '', regex=True).replace(r'@\w+', '', regex=True).replace(r'\n', '', regex=True)
text_to_analyze = text_to_analyze.to_frame().assign(
    order=range(1, len(text_to_analyze)+1, 1)).applymap(str)
text_to_analyze = text_to_analyze.assign(
    prompt=text_to_analyze.order+". \\"+text_to_analyze.text)
prompt = 'Classify the sentiment in these tweets:\n\n' + \
    text_to_analyze.loc[:, 'prompt'].str.cat(sep='\n')

response = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    temperature=0,
    max_tokens=60,
    top_p=1,
    frequency_penalty=0.5,
    presence_penalty=0
)
response
text_to_analyze.iloc[4, 0]
pd.Series(response['choices'][0]['text'].split('\n'))

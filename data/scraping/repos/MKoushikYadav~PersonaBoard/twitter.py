username='koushik15024'
password='Roopa.18'
userlist=['koushik15024','techieMeIndian','Guptaajikaladka']

from tweety import Twitter
app=Twitter('megathon')
app.sign_in(username=username, password=password)
usertweets= app.get_tweets(username=username,pages=1)
a=('\n').join([tweet.text for tweet in usertweets.tweets])

import openai
openai.api_key = "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"

completion = openai.ChatCompletion.create(
model="gpt-3.5-turbo",
messages=[{"role":"user","content":'You will be given some social media posts made by a person. You task is to determine the person\'s personality by analyzing their posts. You need to provide 3-5 positives and/or 3-5 negatives about the person. Here are some strict instructions you must follow: 1. Each Positive/Negative must not be more than 4 words. 2. Avoid duplicate topics. 3. THE FORMAT OF YOUR OUTPUT MUST STRICTLY CONTAIN ONLY THE LIST OF COMMA SEPARATED POSITIVE TOPICS FOLLOWED COMMA SEPARATED NEGATIVE TOPICS, SEPARATED BY NEWLINE. 4. ENSURE THAT NOT EVEN A SINGLE OTHER WORD i.e, NO UN-NECESSARY CHARACTERS(ASTERISKS, HYPHENS, etc) OR EXPLANATIONS ARE INCLUDED. Here is the data: \n'+ a}]
)
# result=bard.get_answer('Can you give me a list of topics this belongs to in the following format (topic\
# topic1\
# topic2\
# so on with the heading being topics: \
# for the following data\n'+a)
print(completion.choices[0].message.content)
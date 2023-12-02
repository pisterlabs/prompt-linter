import os
import openai

testKey = os.environ.get('API_KEY')
#print(testKey)
openai.api_key = testKey
file = open("prompt.txt", "r")
prompt = ""
for a in file.readlines():
    prompt+= a + ""

file = open("characters.txt", 'r')
for i, a in enumerate(file.readlines()):
    completion = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": prompt},
            {"role" : "user", "content" : a}
        ]
        )
    story= completion.choices[0].message.content
    print(i,a,":\n",story,"\n")
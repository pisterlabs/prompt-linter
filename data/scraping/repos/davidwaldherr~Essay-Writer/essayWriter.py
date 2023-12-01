import sys
import pandas as pd
import numpy as np
import openai
openai.api_key ='INSERT API KEY HERE'
from openai.embeddings_utils import get_embedding, cosine_similarity

df = pd.read_csv('Summarize/books/LettersSentences.csv') # automate this later
df['babbage_search'] = df.babbage_search.apply(eval).apply(np.array)


def getBook():
    print("\n\nWhich book are you working with today?")
    print("1. Letters From A Stoic")
    book = input("Input Number: ")
    return book

def getTopic():
    print("Example Essay Topic: Nikola Tesla and his contributions to technology")
    topic = input("Enter a topic: ")
    return topic

def createOutline(topic):
    return openai.Completion.create(
  model="text-davinci-002",
  prompt="Create an outline for an essay about " + topic + ":",
  temperature=0.7,
  max_tokens=256,
  n=1, 
)

def findOutline(topic):
    outlineList = []
    outline = createOutline(topic)
    outline = outline.choices[0].text
    # parse outline into a list by splitting on newlines
    outline = outline.split('\n')
    for i in range(len(outline)):
        if outline[i] != '':
            outlineList.append(outline[i])
    print(outlineList)
    for line in outlineList:
        line = line[3:]
    return outlineList

def getOutlineDecision(outlineList):
    print("Outline:\n")
    for i in range(len(outlineList)):
        print(outlineList[i])
    print("\nWould you like to create a new outline or use an existing outline?")
    print("1. Create a new outline")
    print("2. Use this outline")
    decision = input("\nInput Number: ")
    print('\n')
    return decision

# search through each sentence in the book for the topic
def search_book(df, topicSentence, pprint=True):
    embedding = get_embedding(topicSentence, engine='text-search-babbage-query-001')
    df['similarities'] = df.babbage_search.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('similarities', ascending=False)
    return res

def createInsertion(topicSentence, citation):
    return openai.Completion.create(
  model="text-davinci-002",
  prompt=topicSentence,
  suffix=". " + citation,
  temperature=0.7,
  max_tokens=500,
  n=1, 
)

def finalAPICompletion(bodyParagraph):
    return openai.Completion.create(
  model="curie:ft-personal:lettersfromastoicsummary-2022-07-10-14-49-08",
  prompt=bodyParagraph,
  temperature=0.7,
  max_tokens=256,
  n=1, 
)

models = ["curie:ft-personal:lettersfromastoicauthorvoice-2022-07-10-14-33-24","curie:ft-personal:lettersfromastoicsummary-2022-07-10-14-49-08"]

bookNum = getBook()
if bookNum == 1:
    book = "Letters From A Stoic"
topic = getTopic()
outlineList = findOutline(topic)
outlineDecision = getOutlineDecision(outlineList)
if outlineDecision == 1:
    sys.exit()
quotes = []
citation = []
topicSentences = []
for line in outlineList:
    topicSentences.append(line[:3])
    res = search_book(df, line)
    print("\nTopic Sentence: " + line + "\n")
    for i in range(5):
        print(str(i+1) + ". " + str(res.iloc[i]['sentence']))
    num = int(input("\nInput Number: "))
    num = num - 1
    selection = str(res.iloc[num]['sentence'])
    quotes.append(selection)
    citation.append('"' + str(res.iloc[num]['sentence']) + '" (Chapter: ' + str(res.iloc[num]['chapter']) + ').')
    # print(line + ": " + selection + "\n")
# now you have the quotes, citation, and topic sentences in lists
insert = []
for i in range(len(quotes)):
    insert.append(str(createInsertion(topicSentences[i],citation[i])))
# now you have the insertions in a list
# create a new list with the topic+insertion+citation for a final completion
final = []
finalCompletion = []
for i in range(len(topicSentences)):
    final.append(topicSentences[i] + insert[i] + citation[i])
    finalCompletion.append(finalAPICompletion(final[i]).choices[0].text)
for i in range(len(final)):
    print(topicSentences[i] + insert[i] + citation[i]+ str(finalCompletion[i]))
    print('\n')


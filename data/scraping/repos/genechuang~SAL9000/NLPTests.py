# RAKE https://github.com/fabianvf/python-rake
# howto: https://towardsdatascience.com/extracting-keyphrases-from-text-rake-and-gensim-in-python-eefd0fad582f

import RAKE
from rake_nltk import Rake
import spacy
import pytextrank
import requests
import time
from datetime import datetime
import os
import openai
#TODO: load openai key from GCP Secrets Manager
openai.api_key = "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"



TEST_STRINGS = [
#    "Anyone here use Copper CRM at their company? I’m working with two sales consultants (one is used to Salesforce and the other is used to Hubspot). I personally liked Copper cause it sits on top of Gmail. I’d rather use what the salespeople want to use, but in this case there’s no consensus lol.",
#    "Are there any opinions on accounting systems / ERP's? We're using SAP Business One (aka Baby SAP) and need to upgrade to something a bit more full featured. Personally I find the SAP consulting ecosystem rather abysmal in terms of talent, looking at netsuite as an alternative but curious to know what others are using / we should be looking at.",
#    "Likely this was asked before - as my company is growing, I'm looking for a CTO support group. This Slack is great - but I know there are companies out there organizing 6-8 people pods that meet once a month, and create a tight sharing group and more clear problem-solving support from peers that are going through similar struggles. Anybody here part of such groups that they would recommend? (generally looking for NYC in case we go back to work and meet in person in the second half of this year).",
#    "anyone know of a crawler as a service tool that they've used in the past? What i'd like to do is feed it a list of URLs, and have it dump out data for me in S3 that i can ingest to make decisions on",
#    "Advice on Large Buttons for Arduino/Raspberry Pi Reflex Game I’m looking to create an outdoor game for my kids based on an excercise activity recommended in a book. It’s kind of like whack a mole. I’m looking to place 5-10 buttons around the back yard connected via Bluetooth to a control board (or phone) and buttons are randomly lit up (or announced) and they compete to touch them as quickly as possible and the control board keeps track of their score. The problem is I can’t find buttons like this that aren’t really small. And ideally they’d also have some lights attached. The old Amazon Dash ones would have worked really well and were only $5 (before rebate) so I figured this would be much easier to find.  Any builder parents out there with advice for an Arduino n00b?"
#    "Any builder parents out there with advice for an Arduino n00b"
     "This has been asked a few times on here already, but curious if anyone has developed any strong opinions since the last time it was asked. What has worked the best for your front end teams in E2E testing React Native apps? Appium? Detox?"    
    ]

#TEST_STRINGS = ["What is redis vs mongodb?"]
TEST_USER = 'U5FGEALER' # Gene
TEST_TS = '1616650666.157000'
TEST_CHANNEL_ID = 'GUEPXFVDE' #test

STOPWORDS_LIST=RAKE.SmartStopList()
RAKE_OBJECT = RAKE.Rake(RAKE.SmartStopList())

COMMON_WORDS_3K = {''}
COMMON_WORDS_3K_FILE = open('3kcommonwords.txt')
with COMMON_WORDS_3K_FILE as reader:
    for this_word in reader:
        this_word = this_word.rstrip()
        COMMON_WORDS_3K.add(this_word)
print("3K set has : ", len(COMMON_WORDS_3K))

def RAKEPhraseExtraction(extractString):
    return RAKE_OBJECT.run(extractString)

# Return reverse order List of 2nd element
def sortList(list):
    # key is set to sort using second elemnet of
    # sublist lambda has been used
    list.sort(key = lambda x: x[1])
    list.reverse()
    return list

# Return the top weighed Phrases from RAKE of stringArray
def extractTopPhrasesRAKE(stringArray, removeCommonWords):
    raked = RAKEPhraseExtraction(stringArray)
    print("Raked results: ", raked)
    returnList = []
    for i in raked:
        isCommon = i[0] in removeCommonWords
        if isCommon:
            continue
        returnList.append(i)

    returnList = sortList(returnList)
    print("Raked results remove common words: ", returnList)
    return returnList

#RAKE-NLTK
def RAKENLTKPhaseExtraction(extractString):
    r = Rake() # Uses stopwords for english from NLTK, and all punctuation characters
    r.extract_keywords_from_text(extractString)
    return r.get_ranked_phrases()[0:10]

# Deployed to Google CLoud local - run from repo root dir:
# gcloud functions deploy keyphraseExtraction --runtime python39 --trigger-http --allow-unauthenticated --project=sal9000-307923 --region=us-west2
def keyphraseExtraction(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <https://flask.palletsprojects.com/en/1.1.x/api/#flask.Flask.make_response>`.
    """
    rakeme=''
    returnjson = 0
    request_json = request.get_json()
    if request_json and 'challenge' in request_json:
        #Slack url_verification challenge https://api.slack.com/events/url_verification
        return request_json['challenge']
    elif request.args and 'message' in request.args:
        returnjson = 'returnjson' in request.args
        rakeme = request.args.get('message')
    elif request_json and 'message' in request_json:
        returnjson = request_json['returnjson']
        rakeme =  request_json['message']
    else:
        rakeme = ''
    
    keyPhrases = extractTopPhrasesRAKE(rakeme, COMMON_WORDS_3K)
    return str(keyPhrases)


# Main for commandline run and quick tests
if __name__ == "__main__":

#    print('openai: ')
#    print(openai.Completion.create(engine="text-davinci-001", prompt="Say this is a test", max_tokens=6))

    for extractme in TEST_STRINGS:
 
        response = openai.Completion.create(
            engine="text-davinci-001",
            prompt="Extract keywords from this text:\n\n" + extractme, 
            temperature=0.3,
            max_tokens=60,
            top_p=1,
            frequency_penalty=0.8,
            presence_penalty=0
            )
        print('OpenAI:', response.get("choices"))

#        print('Raking:', extractme)
#        raked = extractTopPhrasesRAKE(extractme, COMMON_WORDS_3K)

#        print('Rake_NLTK results:', RAKENLTKPhaseExtraction(extractme))

#        print('PyTextRanking: ', extractme)
        # load a spaCy model, depending on language, scale, etc.
#        nlp = spacy.load("en_core_web_sm")
        # add PyTextRank to the spaCy pipeline
#        nlp.add_pipe("textrank", last=True)
#        doc = nlp(extractme)
        # examine the top-ranked phrases in the document
#        print('PyTextRank:', doc._.phrases)
#        pytextrankresults = ''
#        for p in doc._.phrases:
#            pytextrankresults += "(" + p.text + ' , ' + str(p.rank)[0:4] + ") "
#            print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
        #    print(p.chunks)
#        print('PyTextRank:', pytextrankresults + "\n")


   
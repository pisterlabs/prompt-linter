import openai
import config
import streamlit as st
openai.api_key = config.openAI
def getNewsFromKeyword(keyword, max):
    qStr = f"""
    {{
        "$query": {{
            "$and": [
                {{
                    "keyword":  "{keyword}",
                    "keywordLoc": "title"
                }},
                {{
                    "locationUri": "http://en.wikipedia.org/wiki/India"
                }},
                {{
                    "dateStart": "2023-05-23",
                    "dateEnd": "2023-05-30",
                    "lang": "eng"
                }}
            ]
        }},
        "$filter": {{
            "dataType": [
                "news",
                "pr",
                "blog"
            ]
        }}
    }}
    """
    q = QueryArticlesIter.initWithComplexQuery(qStr)
    # change maxItems to get the number of results that you want
    return q.execQuery(er, max)


def getNewsFromLink(link):
    response = analytics.extractArticleInfo(link)
    print("News From Link FETCHED! ")
    return response



    
   
    
    return response.choices[0].message.content


def getSentiment(body):
    print("Analysing Sentiment...")
   
    prompt =" Return only the sentiment polarity score for the text: \n"+ str(body)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
             {"role": "system", "content": "You are tasked with generating sentiment polarity score for the given text in the scale -1 to 1. Return only the polarity score and if sentiment cannot be analysed return 0"}
            ,{"role": "user", "content": prompt}
                  ]
        
    )
    print(response.choices[0].message.content)
    sentiment = response.choices[0].message.content
    return sentiment


def generateSummary():
    # data = {}  # Initialize data as an empty dictionary
    prompt = """
    """
    
    
    response = openai.Completion.create(
        engine="text-davinci-003",
         prompt = f"""You are a helpful assistant that gives the sentiment polarity by analyzing the title and description of the youtube video in the scale -1 to 1 with the reasoning for the given score.Format the output as JSON with the following keys:summary:string,polarity:float, reason:string.

ignore chunks of text that contain the following:
- social media links
- shopping links
- equipment information
- copyrights
- music used
- sponsors
- discount and offers

use the given text:
{prompt}
""",

        max_tokens=200,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=1,
        stop=None
        
    )
    
    summary = response.choices[0].text.strip()
    return summary


print(generateSummary())

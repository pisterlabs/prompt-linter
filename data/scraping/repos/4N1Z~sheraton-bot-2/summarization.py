import cohere
import streamlit as st
import os
import requests
import webbrowser
from bs4 import BeautifulSoup

 

#co = cohere.Client(os.environ["COHERE_API_KEY"]) 
def formattingForSummarizer(text):
    for each in text :
        if (each == "'") :
            text = text.replace(each, "")
        if(each == "`"):
            text = text.replace(each, "")    
    
    text = text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    return text


def summarizer (text):

    CleanText = formattingForSummarizer(str(text))
    # prompt_template = "Summarize the given content into 5 paragraph response of `MORE THAN 200 WORDS` each. The content is : " + ' ' + CleanText 
    summarizer_prompt  = "You are the manager of a hotel and you're task is to summarize the given content: that is the details of booking into the format needed for billing.. "

    response = co.summarize( 
          text=CleanText,
          length='long',
          format='bullets',
          model='summarize-xlarge',
          additional_command= summarizer_prompt,
          temperature=0.3,
        ) 
    print(response.summary)
    return response.summary

def generateKBase(largeData):

    rqdFormat =  [
        {
        "title": " ",
        "snippet": " "
        },
    ]
    FormatPrompt = "You should extract the given details text: " + largeData +" into this  \n: "+ " "+ str(rqdFormat) + "\n JSON FORMAT and populate the corresponding values from text : The snippet can contain the large amount of tokens.Don't shortent the content" 
    response = co.generate(
        # text,
        model='command-nightly',
        prompt=FormatPrompt,
        temperature=0.3,
        return_likelihoods =  None,
        # finish_reason= COMPLETE,
        # token_likelihoods= None,
    )
    print(response.generations[0].text)
    # sendAPIReg(response.generations[0].text)




def main():
    text = """The customer wants to book a 1-bedroom suite for 2 days
The check-in date is from 13th September
The suite costs ₹12600 and comes with a king bed, executive lounge access, a shower/tub combination, and amenities like high-speed internet and 2 TVs.
Nothing was mentioned about extras."""

    # generateKBase(data)
    # generateDetails(text)

# Add the summary
if __name__ == "__main__":
    main()


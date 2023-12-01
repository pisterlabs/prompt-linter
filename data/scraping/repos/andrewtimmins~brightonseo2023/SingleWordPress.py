#
# BrightonSEO April 2023, Wordpress GPT example.
#

import requests
import base64
import openai
from bs4 import BeautifulSoup 
import warnings
warnings.filterwarnings("ignore")

# Configurable options
openai.api_key = "***key here***"
wpUser = "***user name here***"
wpPaassword = "**passwordhere***"
wpURL = 'https://**url here***/wp-json/wp/v2/posts/'
newsURL = "http://feeds.bbci.co.uk/news/rss.xml"
	
# Create the authentication header for Wordpress
wpCredentials = wpUser + ":" + wpPaassword
wpToken = base64.b64encode(wpCredentials.encode())
wpHeader = {'Authorization': 'Basic ' + wpToken.decode('utf-8'), 'Content-Type': 'application/json'}

# Fetch the 10 latest news article
xmlResponse = requests.get(newsURL)
xmlSoup = BeautifulSoup(xmlResponse.text, "html.parser")	
newsArticles = xmlSoup.find_all('item')
current_item=0
for item in range(10):
	articleURL = newsArticles[current_item].guid.text
	articleHeadLine = newsArticles[current_item].title.text
	current_item+=1

#print(newsArticles[0].title.text)
#print(newsArticles[0].guid.text)

# Get the first news article in the list, we're not worrying about the others in this example
articleResponse = requests.get(newsArticles[0].guid.text)
articleSoup = BeautifulSoup(articleResponse.text, "html.parser")
articleScrapedContent = articleSoup.find_all('p')
articleText = ""
for p_tag in range(len(articleScrapedContent)):
		if p_tag < 0:
			pass
		else:
			articleText = articleText + articleScrapedContent[p_tag].text

#print(articleText)

# Build the completion based on the extracted content for the content
contentCompletion = openai.ChatCompletion.create(
  model = 'gpt-3.5-turbo',
  messages = [
     {"role": "user", "content": "Rewrite the following text to be more concise: " + articleText}
  ],
  temperature = 0
)

# Place our new content into the variable wpPostContent
wpPostContent = contentCompletion['choices'][0]['message']['content']

# Build the completion based on the extracted content for the content
titleCompletion = openai.ChatCompletion.create(
  model = 'gpt-3.5-turbo',
  messages = [
     {"role": "user", "content": "Rewrite the following text to be more concise: " + newsArticles[0].title.text}
  ],
  temperature = 0
)

# Place our new slug into the variable wpSlug, transforming it to be URL safe
wpSlug = titleCompletion['choices'][0]['message']['content']
wpSlug = wpSlug.lower()
wpSlug = wpSlug.replace(" ", "-")

#print(titleCompletion['choices'][0]['message']['content'])
#print(wpPostContent)
#print(wpSlug)

# Send the new post to Wordpress in a draft state
wpData = {
'title' : titleCompletion['choices'][0]['message']['content'],
'status': 'draft',
'slug' : wpSlug,
'content': wpPostContent
}

wpResponse = requests.post(wpURL,headers=wpHeader, json=wpData)
print(wpResponse)

import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from WebActionIdentifier import WebActionIdentifier
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from DissolveElementChain import DissolveElementChain
from DissolveNavigatorChain import DissolveNavigatorChain

# load dotenv
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

#base template
elementSelectorTemplate = """
You are an intelligent ecommerce site navigator who is helping a user navigate a website. You will be given a query by
the user about what he wants to achieve on the website for example: add a product to his cart, or return their last order, or
filter the products by price. You can also use the fetch_html_code tool to fetch the html code of the page the user is currently viewing.
Your job is to provide the element of the page on which click action will be taken.
"""


elementSelectorPrompt = PromptTemplate(
    input_variables=["query","url"],
    template=elementSelectorTemplate + "User query is: {query}? User is currenlty on this url: {url}",
)

navigationTemplate = """
You help the user to navigate from the current webpage url to the required webpage url using the sitemap. For example: if the current user webpage url is "https://www.logitech.com/en-in" and the user says
that they want to see keyboards which are compatible with iPad then you have to return the url of the page
"http://www.logitech.com/en-in/products/keyboards.html". You have to strictly respond with a url ONLYYYYY where the user, 
this is a non-negotiable. Only provide me with the url no other text.
For example just resply with: http://www.logitech.com/en-in/products/keyboards.html
"""

navigationPrompt = PromptTemplate(
    input_variables=["query","url"],
    template = navigationTemplate + "User query is: {query}? User is currenlty on this url: {url}",
)

llm = ChatOpenAI(openai_api_key=openai_api_key,model_name='gpt-3.5-turbo-16k-0613',temperature=0)

# class DissolveAgent:
#     def __init__(self, query: str, url: str):
#         self.query = query
#         self.url = url

    # def _call(self):
navigationChain = DissolveNavigatorChain(llm=llm,prompt=navigationPrompt,verbose=True)
elementSelectorChain = DissolveElementChain(llm=llm,prompt=elementSelectorPrompt,verbose=True)

    # for dev only
query = "show me all keyboards which work with windows and linux"
query = "show me webcams with HD recording"
url = "https://www.logitech.com/en-in"

print("User on this url: ",url)
print("User query: ",query)

newUrl = navigationChain.run({
    'query':query,
    'url':url
})

elementList = elementSelectorChain.run({
    'query':query,
    'url':newUrl
    })

output = {
    'redirectUrl':newUrl,
    'selectorString':elementList
}

print(output)

# return output
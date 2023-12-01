from googlesearch import search
import requests
from bs4 import BeautifulSoup
import openai
import time
import os
from dotenv import load_dotenv
load_dotenv()


class WebsiteScrater:                     #Sraper+Rater=Srater :)
    API_KEY = os.getenv("API_KEY")  # API KEY for the google cloud project
    ID = os.getenv("ID")            # ID for search engine 

    #Make sure you pass the topic during instantiation
    def __init__(self,topic):
        self.topic = topic
        print(self.API_KEY,self.ID)

    #This function can be used to reset the topic
    def reset_topic(self,topic):
        self.topic = topic

    #This function takes a website's content and rates it on a scale of 1-5 using prompt engineering 
    def rate_content(self,content,topic,link):
        openai.api_key = os.getenv("OPENAI_API")
        prompt = f'''Topic:{topic} .On a scale of 1 to 5, rate the website's content. Deduct points for websites that primarily promote a course on the topic rather than providing informative content about the subject matter(note that if it provides some content then it's okay).You should only say the rating(in integer) and nothing else[important]-> '''+content[:15000]
        try:
            #return 5   # adding this to reduce the credit usage for now
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = [ {"role":"user","content":prompt}]
            )
            #print("link:"+link)
            #print(response)
        except Exception as e:
            print(e)
            return "Limit Exceeded"
        return response
    
    #This function extracts the only the text content from a website 
    def extract_content(self,data):
        tags = data.find_all(['p','h1','h2','h3','h4','h5','h6','li','span'])
        content = ""
        for tag in tags:
            content+=tag.get_text().strip()
        #print(len(content))
        return content
    
    #This function extracts the soups for all the website links provided
    def extract_soups(self,links):
        soups = []
        for link in links:
            print(link)
            req = requests.get(link).text
            soup = BeautifulSoup(req,"lxml")
            soups.append(soup)
            #print(soup)
        return soups
    
    #This function gives the top 10 search results relevant to the topic provided
    def get_links(self):
        user_input='what is '+self.topic+"?"
        self.query = user_input
        url='https://www.googleapis.com/customsearch/v1'
        params={
            'q':user_input,
            'key':self.API_KEY,
            'cx':self.ID
        }     
        response = requests.get(url, params=params)
        result=response.json()
        links = []
        #print(result)
        for item in result['items']:
            links.append(item.get('link', ''))
        return links

    #This function returns the websites with their rating in a list ->[[website1,rating],[website2,rating],....]
    def get_results(self):
        #results = search(self.query,num_results=2)
        #links = [link for link in results]
        links = self.get_links()
        soups = self.extract_soups(links=links)
        contents = [self.extract_content(soup) for soup in soups]
        websites_with_rating = []
        for i in range(3):
            response = self.rate_content(contents[i],self.query,links[i])
            rating = 3 if type(response)==str or type(response)==int else int(response["choices"][0]["message"]["content"])
            #print(links[i],rating,response)
            websites_with_rating.append([links[i],rating])
            #time.sleep(5)
        ratings_json = {"data":websites_with_rating}
        return ratings_json
        #print(len(links))

if __name__=='__main__':
    ws = WebsiteScrater("python")
    print(ws.get_results())
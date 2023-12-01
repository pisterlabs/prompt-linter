from dotenv import load_dotenv
import tweepy
import os 
import logging
import requests
from bs4 import BeautifulSoup
import random
from langchain import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain import LLMChain

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
handler = logging.FileHandler(filename="twitter-bot-blog.log")
logger.addHandler(handler)

MAX_TRIES = 5

def main():
	# sacar lista de posts    
	url = "https://www.sensiocoders.com/blog/"  
	response = requests.get(url)
	soup = BeautifulSoup(response.content, 'html.parser')
	posts = soup.find_all('a', href=lambda href: href and "/blog/" in href) 
	# setup langcahin
	model = 'gpt-3.5-turbo-16k' 
	llm = OpenAI(model_name=model, temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'], max_tokens=256)
	template = """Tu tarea es leer el siguiente post y generar un tweet educativo en castellano que haga crecer el número de seguidores de la cuenta.
Dado el siguiente post, genera un tweet de máximo 200 caracteres en castellano.
El tweet debe ser educativo y accesible para diferentes niveles de expertos.
Responde siempre en español.
Utiliza hashtags apropiados para llegar a más personas.

POST:

{post} 

TWEET:"""
	prompt = PromptTemplate(template=template, input_variables=['post'])
	llm_chain = LLMChain(prompt=prompt, llm=llm)
	# cliente de twitter
	client = tweepy.Client(
	    bearer_token=os.getenv("BEARER_TOKEN"),
	    access_token=os.getenv("ACCESS_TOKEN"),
	    access_token_secret=os.getenv("ACCESS_TOKEN_SECRET"),
	    consumer_key=os.getenv("API_KEY"),
	    consumer_secret=os.getenv("API_KEY_SECRET")
	)
	# generar tweet
	i = 1
	while i <= MAX_TRIES:
		try:
			# elegir post al azar
			post = random.choice(posts)
			# leer post
			post_url = post['href'].split('/')[-1]
			post_response = requests.get(url + post_url)
			post_soup = BeautifulSoup(post_response.content, 'html.parser')
			content = post_soup.find('div', class_="post").text
			# generar tweet
			tweet = llm_chain.run(content)
			tweet += " https://www.sensiocoders.com/blog/" + post_url
			assert len(tweet) <= 280
			print(tweet)
			client.create_tweet(text=tweet)
			return
		except:
			continue
		finally:
			i += 1
	print("No se pudo generar un tweet :(")
	return
 
if __name__ == "__main__":
	load_dotenv()
	main()
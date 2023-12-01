import os
import sys

import constants
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI


os.environ["OPENAI_API_KEY"] = constants.OPENAI_APIKEY

query = sys.argv[1]
print(query)

print('Loading Data...')

"""
# Populate text file with champion data from Riot API
import cassiopeia as cass

print('Loading Champions...')
champions = cass.get_champions(region="NA")

static = open("static_data.txt", "w")
for champ in champions:
    static.write("New Champion: \n")
    static.write(champ.name + '\n')
    static.write(champ.title + '\n')
    static.write(champ.lore + '\n')
    static.write("Ally Tips: " + ' '.join(champ.ally_tips) + '\n')
    static.write("Enemy Tips: " + ' '.join(champ.enemy_tips) + '\n')

print('Loading Data Complete.')

print('Champions: ' + str(len(champions)))
"""
# Commented out because we don't have to run it everytime 
# since we have the data stored in a text file
"""
# Populate text file with champion bios from Official League of Legends Website
import requests
from bs4 import BeautifulSoup

url_names = open("url_names.txt", "r")
champ_names = url_names.readlines()

lore_file = open("lore.txt", "w")
for champ_name in champ_names:
    url = f"https://universe.leagueoflegends.com/en_US/story/champion/{champ_name[:-1]}/"
    print(champ_name[:-1])
    page = requests.get(url)
    print(page.status_code)
    soup = BeautifulSoup(page.content, 'html.parser')

    lore_file.write("\n Next Champion: " + champ_name + '\n')

    title = soup.find('meta', property="og:title")['content']
    lore_file.write(title + '\n')

    desc = soup.find('meta', property="og:description")['content'] 
    lore_file.write(desc + '\n')
"""

load_lore = TextLoader('./lore.txt')
load_static = TextLoader('./static_data.txt')

index = VectorstoreIndexCreator().from_loaders([load_lore, load_static])

print(index.query(query, llm=ChatOpenAI()))
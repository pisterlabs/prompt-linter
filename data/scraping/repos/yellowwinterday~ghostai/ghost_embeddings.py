import os
import sys
import re
import json
import requests # pip install requests
import jwt	# pip install pyjwt
import configparser
import pandas as pd
import numpy as np
import openai
import logging

from datetime import datetime as date
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

config = configparser.ConfigParser()
config.read('./.env')
key = ""
try:
    key = config['BASIC']['GHOST_ADMIN_API_KEY']
except KeyError as e:
    print("Not finding key in config. Try environment.")

if len(key) == 0:
    if 'GHOST_ADMIN_API_KEY' not in os.environ:
        print("Missing GHOST_ADMIN_API_KEY. Please set the GHOST_ADMIN_API_KEY environment variable before running the script. You can find it by following instructions here: https://ghost.org/docs/admin-api/")
        exit(0)
    else:
        key = os.environ.get('GHOST_ADMIN_API_KEY')

print("GHOST_ADMIN_API_KEY ready")
openai_key = ""
try:
    openai_key = config['BASIC']['OPENAI_API_KEY']
except KeyError as e:
    print("Not finding key in config. Try environment.")

if len(openai_key) == 0:
    if 'OPENAI_API_KEY' not in os.environ:
        print("Missing OPENAI_API_KEY. Please set the OPENAI_API_KEY environment variable before running the script. You can find it by following instructions here: https://platform.openai.com/account/api-keys")
        exit(0)
    else:
        openai_key = os.environ.get('OPENAI_API_KEY')

print("OPENAI_API_KEY ready")

openai.api_key = openai_key

url = config['BASIC']['GHOST_SITE_URL']
log_path = config['BASIC']['LOG_PATH']
output_path = config['BASIC']['EMBEDDING_OUTPUT_PATH']
max_related_count = int(config['BASIC']['MAX_RELATED_BLOG_COUNT'])

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print("Directory "+str(output_path)+" created.")
#else:
#    print("Directory "+str(output_path)+" already exists.")

if not os.path.exists(log_path):
    os.makedirs(log_path)
    print("Directory "+str(log_path)+" created.")
#else:
#    print("Directory "+str(log_path)+" already exists.")

# Configure logging
logging.basicConfig(
    filename=str(log_path)+'/embedding.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def fetchBlogPage(page):
    # Split the key into ID and SECRET
    id, secret = key.split(':')

    # Prepare header and payload
    iat = int(date.now().timestamp())

    header = {'alg': 'HS256', 'typ': 'JWT', 'kid': id}
    payload = {
        'iat': iat,
        'exp': iat + 5 * 60,
        'aud': '/admin/'
    }

    # Create the token (including decoding secret)
    token = jwt.encode(payload, bytes.fromhex(secret), algorithm='HS256', headers=header)

    # Make an authenticated request to list all posts
    headers = {'Authorization': 'Ghost {}'.format(token)}
    geturl = url+'ghost/api/admin/posts/?page='+str(page)

    return requests.get(geturl, headers=headers)

def getAllBlogIDs():
    files = os.scandir(output_path)
    blog_ids = []
    for file in files:
        if re.search('blog-([\d\w]*)-embedding.csv', file.name):
            blog_id = re.search('blog-([\d\w]*)-embedding.csv', file.name).group(1)
            blog_ids.append(blog_id)
    return blog_ids


def generateEmbeddingsForAllBlogs():
    response = fetchBlogPage(1)

    if int(response.status_code) != 200:
        sys.exit('Failed to load blog list, please check ./.env to make sure all keys are set.')

    total_page = int(response.json()['meta']['pagination']['pages'])
    total_blog = int(response.json()['meta']['pagination']['total'])
    print(str(total_page)+' pages to load')
    print(str(total_blog)+' blogs to load')

    for page in range(1,total_page+1):
        response = fetchBlogPage(page).json()
        posts = response['posts']
        for post in posts:
            postContent = ""
            id = post['id']
            print("Parsing blog-"+str(id))
            logging.info("Parsing blog-"+str(id))
            embedding_file_path = output_path+"/blog-"+str(id)+"-embedding.csv"
            if not os.path.exists(embedding_file_path):
                title = post['title']
                postContent = postContent + str(title) + '\n'

                if post['mobiledoc'] == None:
                    print("Blog failed to parse content due to missing 'mobiledoc' data.")
                    logging.info("Blog failed to parse content due to missing 'mobiledoc' data.")
                    logging.info("Blog post content:"+str(post))
                    print("Skip to next blog.")
                    continue

                try:
                    mobiledoc = json.loads(post['mobiledoc'])
                except Exception as e:
                    #Do a simple retry
                    print('Blog failed to parse mobiledoc data due to error:')
                    print(e)
                    logging.info('Blog failed to parse mobiledoc data due to error:')
                    logging.info(e)
                    print("Skip to next blog.")
                    continue


                cards = mobiledoc['cards']
                for card in cards:
                    if card[0] == 'toggle':
                        headContent = card[1]['heading']
                        headContent = headContent.replace('<p>','')
                        headContent = headContent.replace('</p>','\n')
                        content = card[1]['content']
                        content = content.replace('<p>','')
                        content = content.replace('</p>','\n')
                        postContent = postContent + headContent + '\n' + content + '\n'
                    elif card[0] == 'html':
                        htmlString = card[1]['html']
                        postContent = postContent + htmlString + '\n'

                sections = mobiledoc['sections']
                for section in sections:
                    if int(section[0]) == 1:
                        paragraph = list(section[2])
                        if len(paragraph) > 0:
                            if len(paragraph[0]) > 3:
                                postContent = postContent + str(paragraph[0][3])

                html = ""
                if 'html' in post:
                    html = json.loads(post['html'])
                    postContent = postContent + str(html)

                if len(postContent) > 0:
                    try:
                        embedding = get_embedding(postContent, engine='text-embedding-ada-002')
                    except Exception as e:
                        #Do a simple retry
                        try:
                            embedding = get_embedding(postContent, engine='text-embedding-ada-002')
                        except Exception as e:
                            print('Blog failed to convert due to error:')
                            print(e)
                            print('ID:'+id+'\nTitle:'+title)
                            print('Post content:'+str(postContent))
                            print('Original mobiledoc:'+str(mobiledoc))
                            print('Original cards:'+str(cards))
                            print('Original html:'+str(html))
                            print('Original post:')
                            print(post)
                            logging.info('Blog failed to convert due to error:')
                            logging.info(e)
                            logging.info('ID:'+id+'\nTitle:'+title)
                            logging.info('Post content:'+str(postContent))
                            logging.info('Original mobiledoc:'+str(mobiledoc))
                            logging.info('Original cards:'+str(cards))
                            logging.info('Original html:'+str(html))
                            logging.info('Original post:')
                            logging.info(post)
                            continue


                    newdf = pd.DataFrame({"blog_id":[id],"title":[title],"embedding":[embedding]})
                    newdf.to_csv(embedding_file_path,index=None)
                    print("blog-"+str(id)+" embedding generated")
                    logging.info("blog-"+str(id)+" embedding generated")
                    #logging.info('Embedded content:'+str(postContent))
                else:
                    print("blog-"+str(id)+" content failed to load. Skip to next blog.")
                    logging.info("blog-"+str(id)+" content failed to load. Skip to next blog.")
                    logging.info('ID:'+id+'\nTitle:'+title)
                    logging.info('Post content:'+str(postContent))
                    logging.info('Original mobiledoc:'+str(mobiledoc))
                    logging.info('Original cards:'+str(cards))
                    logging.info('Original html:'+str(html))
                    logging.info('Original post:')
                    logging.info(post)
            #else:
                #print("blog-"+str(id)+" embedding existed")


generateEmbeddingsForAllBlogs()

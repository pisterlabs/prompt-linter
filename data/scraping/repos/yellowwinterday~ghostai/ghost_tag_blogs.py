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
blog_tag_count = int(config['BASIC']['BLOG_TAG_COUNT'])

prompt = "Please find "+str(blog_tag_count)+" tags of the following paragraphs, separated by commas, each tag with only one word. Paragraph:"
prompt += '\n'

print("Prompt: " + prompt)


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
    filename=str(log_path)+'/tagging.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

reset_all = False
if len(sys.argv) > 1:
    if str(sys.argv[1]).upper() == "RESET":
        reset_all = True


if reset_all:
    print("Resetting all tags")
    logging.info("Resetting all tags")
else:
    print("Generate tags for new blogs")
    logging.info("Generate tags for new blogs")

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

def tagContent(prompt, content):
    prompt += content
    if len(prompt) > 10000:
        prompt = prompt[:10000]

    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "user", "content": prompt},
      ],
      temperature=0.4,
      max_tokens=1000,
      top_p=1,
      frequency_penalty=0.2,
      presence_penalty=1.6
    )

    result = completion.choices[0].message['content']
    result = result.lstrip()
    logging.info('Generate Tag Result:'+str(result))
    #print(result)


    tags = result.replace("\n",",")
    tags = tags.split(",")
    #print(tags)

    qualified_tags = []
    for tag in tags:
        tag_words = tag.lstrip()
        if len(tag_words) > 0:
            wordscount = tag_words.split(" ")
            if len(wordscount) <= 2:
                qualified_tags.append(tag_words.upper())


    #print(qualified_tags)

    #final_tags = ', '.join(qualified_tags)
    #print(final_tags)
    return qualified_tags

def ghost_update_public_tags(site_url,blog_id,tags):

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

    # Make an authenticated request to create a post
    geturl = site_url+'ghost/api/admin/posts/'+blog_id
    headers = {'Authorization': 'Ghost {}'.format(token)}

    try:
        r = requests.get(geturl, headers=headers)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        # catastrophic error. bail.
        raise SystemExit(e)

    #print(r.json()['posts'][0]['updated_at'])
    updated_at = r.json()['posts'][0]['updated_at']
    original_tags =r.json()['posts'][0]['tags']

    # Make an authenticated request to edit an item
    headers = {'Authorization': 'Ghost {}'.format(token)}
    editurl = site_url+'ghost/api/admin/posts/'+blog_id+'/?formats=mobiledoc%2Clexical'


    tag_dict_array = []

    for original_tag in original_tags:
        tag_dict = {'name':str(original_tag['name']),'slug':str(original_tag['slug'])}
        if original_tag['name'][0] == '#':
            tag_dict_array.append(tag_dict)

    for original_tag in original_tags:
        tag_dict = {'name':str(original_tag['name']),'slug':str(original_tag['slug'])}
        if original_tag['name'][0] != '#':
            logging.info("Find existing tags:"+original_tag['name'])
            if reset_all:
                logging.info('over ride new tags')
            else:
                tag_dict_array.append(tag_dict)

    for tag in tags:
        tag_dict = {'name':str(tag),'slug':str(tag)}
        duplicate_tag = False
        for original_tag in tag_dict_array:
            if original_tag['name'] == str(tag):
                logging.info("Find duplicated tags:"+original_tag['name'])
                duplicate_tag = True
        if duplicate_tag == False:
            tag_dict_array.append(tag_dict)


    logging.info('Updated tags:'+str(tag_dict_array))

    body = {
        "posts":[
            {
                "tags":tag_dict_array,
                "updated_at":updated_at
            }
        ]
    }

    try:
        r = requests.put(editurl, json=body, headers=headers)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Update tag failed due to request: " + str(e))
        return False
    except Exception as e:
        print("Update tag failed: " + str(e))
        return False
    #print(r.json()['posts'][0]['tags'])
    return True

def generateAndUpdateTagsForAllBlogs():
    response = fetchBlogPage(1)

    if int(response.status_code) != 200:
        sys.exit('Failed to load blog list, please check ./.env to make sure all keys are set.')

    total_page = int(response.json()['meta']['pagination']['pages'])
    total_blog = int(response.json()['meta']['pagination']['total'])
    print(str(total_page)+' pages to load')
    print(str(total_blog)+' blogs to load')

    count = 0
    for page in range(1,total_page+1):
        response = fetchBlogPage(page).json()
        posts = response['posts']
        for post in posts:
            postContent = ""
            id = post['id']
            tagging_file_path = output_path+"/blog-"+str(id)+"-tags.txt"
            needs_tagging = False
            if reset_all or not os.path.exists(tagging_file_path):
                needs_tagging = True

            if reset_all and os.path.exists(tagging_file_path):
                print("Blog id:"+post['id']+" tags existed but require reset.")
                logging.info("Blog id:"+post['id']+" tags existed but require reset.")

            if needs_tagging:
                title = post['title']
                mobiledoc = json.loads(post['mobiledoc'])
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

                try:
                    qualified_tags = tagContent(prompt, postContent)
                except Exception as e:
                    try:
                        qualified_tags = tagContent(prompt, postContent)
                    except Exception as e:
                        print('Blog failed to convert due to error:')
                        print(e)
                        print('ID:'+id+'\nTitle'+title)
                        print('Post content:'+str(postContent))
                        print('Original mobiledoc:'+str(mobiledoc))
                        print('Original cards:'+str(cards))
                        logging.info('Blog failed to tag due to error:')
                        logging.info(e)
                        logging.info('ID:'+id+'\nTitle'+title)
                        logging.info('Post content:'+str(postContent))
                        logging.info('Original mobiledoc:'+str(mobiledoc))
                        logging.info('Original cards:'+str(cards))
                        continue

                print("blog-"+str(id)+" tags generated: "+str(', '.join(qualified_tags)))
                logging.info("blog-"+str(id)+" tags generated")
                logging.info('Tagging content:'+str(postContent))

                if ghost_update_public_tags(url,id,qualified_tags):
                    print("Updated blog id:"+post['id']+" tags, tags recorded in "+tagging_file_path)
                    logging.info("Updated blog id:"+post['id']+" tags, tags recorded in "+tagging_file_path)
                    f = open(tagging_file_path, "w")
                    f.write(str(', '.join(qualified_tags)))
                    f.close()
                    count += 1
                else:
                    print("Updated blog id:"+post['id']+" tags failed.")
                    logging.info("Updated blog id:"+post['id']+" tags failed.")

            else:
                print("blog-"+str(id)+" tags existed")
    print("Total "+str(count)+" blog tagged.")
    logging.info("Total "+str(count)+" blog tagged.")


generateAndUpdateTagsForAllBlogs()

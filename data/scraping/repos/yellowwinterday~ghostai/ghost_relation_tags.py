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

url = config['BASIC']['GHOST_SITE_URL']
log_path = config['BASIC']['LOG_PATH']
output_path = config['BASIC']['EMBEDDING_OUTPUT_PATH']
max_related_count = int(config['BASIC']['MAX_RELATED_BLOG_COUNT'])


if not os.path.exists(output_path):
    os.makedirs(output_path)
    print("Directory '{output_path}' created.")
#else:
#    print("Directory '{output_path}' already exists.")

if not os.path.exists(log_path):
    os.makedirs(log_path)
    print("Directory '{log_path}' created.")
#else:
#    print("Directory '{log_path}' already exists.")

# Configure logging
logging.basicConfig(
    filename=str(log_path)+'/relation_tags.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def getAllBlogIDs():
    files = os.scandir(output_path)
    blog_ids = []
    for file in files:
        if re.search('blog-([\d\w]*)-embedding.csv', file.name):
            blog_id = re.search('blog-([\d\w]*)-embedding.csv', file.name).group(1)
            blog_ids.append(blog_id)
    return blog_ids

def readEmbedding():
    df = pd.DataFrame({"blog_id":[],"title":[]})
    blog_ids = getAllBlogIDs()

    for blog_id in blog_ids:
        embedding_file_path = output_path+"/blog-"+str(blog_id)+"-embedding.csv"
        if os.path.exists(embedding_file_path):
            print("found embedding for blog-"+str(blog_id))
            blog_df = pd.read_csv(embedding_file_path)
            blog_df['embedding'] = blog_df['embedding'].apply(np.array) #sanity checks, make sure it's an array of strings
            blog_df['embedding'] = blog_df['embedding'].apply(eval) #sanity checks, make sure it's an array of strings
            newdf = pd.DataFrame({"blog_id":str(blog_id),"title":blog_df['title'],"embedding":blog_df['embedding']})
            df = pd.concat([df,newdf],ignore_index=True)
        else:
            logging.info("Missing embedding or blog_id for topic-"+str(blog_id))
            print("Missing embedding or blog_id for topic-"+str(blog_id))
    return df

def ghost_update_interal_tags(url,blog_id,tags):

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
    geturl = url+'ghost/api/admin/posts/'+blog_id
    headers = {'Authorization': 'Ghost {}'.format(token)}

    try:
        r = requests.get(geturl, headers=headers)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logging.info("Get blog tag failed due to http error: " + str(e))
        if e.response.status_code == 404:
            print("Clean up missing blog cache:")
            logging.info("Clean up missing cache:")

            missing_embedding_file_path = output_path+"/blog-"+str(blog_id)+"-embedding.csv"
            if os.path.isfile(missing_embedding_file_path):
                try:
                    os.remove(missing_embedding_file_path)
                    print(f"Cache file '{missing_embedding_file_path}' successfully cleaned.")
                    logging.info(f"Cache file '{missing_embedding_file_path}' successfully cleaned.")
                except OSError as e:
                    print("Error deleting cache file: {e}")

            missing_relation_file_path = output_path+"/blog-"+str(blog_id)+"-relations.csv"
            if os.path.isfile(missing_relation_file_path):
                try:
                    os.remove(missing_relation_file_path)
                    print(f"Cache file '{missing_relation_file_path}' successfully cleaned.")
                    logging.info(f"Cache file '{missing_relation_file_path}' successfully cleaned.")
                except OSError as e:
                    print("Error deleting cache file: {e}")
        else:
            logging.info(f"An HTTP error occurred: {e}")
        return False
    except Exception as e:
        print("Get blog tag failed due to exception: " + str(e))
        return False

    #print(r.json()['posts'][0]['updated_at'])
    updated_at = r.json()['posts'][0]['updated_at']
    original_tags =r.json()['posts'][0]['tags']

    # Make an authenticated request to edit an item
    headers = {'Authorization': 'Ghost {}'.format(token)}
    # DEBUG
    editurl = url+'ghost/api/admin/posts/'+blog_id+'/?formats=mobiledoc%2Clexical'


    tag_dict_array = []
    for original_tag in original_tags:
        tag_dict = {'name':str(original_tag['name']),'slug':str(original_tag['slug'])}
        if original_tag['name'][0] != '#':
            tag_dict_array.append(tag_dict)

    for tag in tags:
        tag_dict = {'name':'#'+str(tag),'slug':str(tag)}
        duplicate_tag = False
        for original_tag in tag_dict_array:
            if original_tag['name'] == '#'+str(tag):
                duplicate_tag = True
        if duplicate_tag == False:
            tag_dict_array.append(tag_dict)

    #print(tag_dict_array)

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
    except requests.exceptions.HTTPError as e:
        print("Update tag failed due to http error: " + str(e))
        if e.response.status_code == 404:
            logging.info("Error: Page not found (404)")
        else:
            logging.info(f"An HTTP error occurred: {e}")
        return False
    except Exception as e:
        print("Update tag failed due to exception: " + str(e))
        return False

    #print(r.json()['posts'][0]['tags'])
    return True

def setupRelationship():
    total_df = readEmbedding()

    files = os.scandir(output_path)

    blog_ids = getAllBlogIDs()

    for blog_id in blog_ids:
        similar_df = total_df
        embedding_file_path = output_path+"/blog-"+str(blog_id)+"-embedding.csv"
        if os.path.exists(embedding_file_path):
            #print("found embedding for blog-"+str(blog_id))
            blog_df = pd.read_csv(embedding_file_path)
            blog_df['embedding'] = blog_df['embedding'].apply(np.array) #sanity checks, make sure it's an array of strings
            blog_df['embedding'] = blog_df['embedding'].apply(eval) #sanity checks, make sure it's an array of strings
            blog_df = pd.DataFrame({"blog_id":str(blog_id),"title":blog_df['title'],"embedding":blog_df['embedding']})
            similar_df["similarities"] = similar_df['embedding'].apply(lambda x: cosine_similarity(x, blog_df['embedding'][0]))
            #print(similar_df)
            similar_df = similar_df.drop(similar_df[(similar_df.blog_id == str(blog_id))].index)
            similar_df = similar_df.drop(columns=['embedding'])
            top_results = similar_df.sort_values("similarities", ascending=False).head(max_related_count)
            #top_title_list = top_results.loc[:, 'title'].tolist()
            logging.info(top_results)
            top_results.to_csv(output_path+"/blog-"+str(blog_id)+"-relations.csv")
            print("Generated relation for blog-"+str(blog_id)+" successful")
            logging.info("Generated relation for blog-"+str(blog_id)+" successful")
            related_ids = top_results['blog_id'].tolist()
            #print(related_ids)
            if ghost_update_interal_tags(url,blog_id,related_ids):
                print("Updated tags for blog-"+str(blog_id)+" successful")
                logging.info("Updated tags for blog-"+str(blog_id)+" successful")
            else:
                #print("Failed to update tags. If this issue persists, please clean up output path and run the script again.")
                logging.info("Failed to update tags. If this issue persists, please clean up output path and run the script again.")
        else:
            print("Missing embedding for blog-"+str(blog_id))
            logging.info("Missing embedding for blog-"+str(blog_id))

setupRelationship()



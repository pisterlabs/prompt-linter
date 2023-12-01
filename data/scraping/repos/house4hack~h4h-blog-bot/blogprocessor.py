import random
import blogbot_utils as bu
import json
import telebot
import openai
import select 
import traceback
import sys

import requests
import base64
import os
import re
from datetime import datetime

import threading
import queue
import time

from jinja2 import Environment, FileSystemLoader
from bs4 import BeautifulSoup
from bs4.element import Comment


import subprocess


def get_random_style():
    with open("templates/style.txt") as f:
        style = f.readlines()
    style = random.choice(style).strip()
    return style



def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(string=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

class BlogProcessorWorker(threading.Thread):
    def __init__(self, q:queue.Queue, config:dict, *args, **kwargs):
        self.q = q
        self.config = config
        with open ("config.json") as f:
            self.config = json.load(f)

        self.bot = telebot.TeleBot(self.config['telegram_bot'])
        super().__init__(*args, **kwargs)

    def run(self):
        while True:
            try:
                tasktype, user_id = self.q.get(block=False)  # 3s timeout
                have_work = True
            except queue.Empty:
                have_work = False
                time.sleep(1)

            if have_work:
                try:
                    if tasktype == "preview":
                        self.process_task(user_id)
                        
                    elif tasktype == "publish":
                        self.publish_task(user_id)

                    elif tasktype == "reload":
                        self.reload(user_id)
                        
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    self.bot.send_message(user_id, "Something went wrong, please try again later:\n {}".format(e))

    def get_prev_posts(self, count:int):
        password = self.config['wordpress_key']
        url = self.config['wordpress_url']
        if url.endswith("/"):
            url = url[:-1]
        category = self.config['wordpress_category']
        href = f"{url}/wp-json/wp/v2/posts?categories={category}"
        user = self.config['wordpress_user']
        credentials = user + ':' + password
        token = base64.b64encode(credentials.encode())


        header = {'Authorization': 'Basic ' + token.decode('utf-8')}
        responce = requests.get(href ,params={'categories':category}, headers=header)
        if responce.status_code != 200:
            raise Exception(f"Error getting posts: {responce.status_code} {responce.text}")
        jj = responce.json()
        jj = jj[:count]
        result = []
        for post in jj:
            text = text_from_html(post['content']['rendered'] )
            title = text_from_html("<html><body><p>"+post['title']['rendered']+ "</p></body></html>")
            #print(title, post['title']['rendered'])
            result.append({"title":title, "contents":text, "date":post['date'][:10]})
        return result





    def process_task(self, user_id : str):
        '''Processes the task'''
        
        bu.set_status(user_id, "Generating")
        conversation = bu.get_conversation(user_id)

        text_list = []

        # make caption strings
        caption_list = []
        i = 1
        for m in conversation['messages']:
            if m['kind'] == "text":
                text_list.append(m['text'])
            elif m['kind'] == 'media':
                if m.get('text',None) is not None and m.get('text','').strip() != '':
                    caption_list.append((i, m['filename'],m['text']))
                    m['slug'] = f"Photo_{i}"
                    i+=1

        bu.save_conversation(user_id, conversation)
        caption_str = "\n".join([f"Photo_{c[0]} : \"{c[2]}\"" for c in caption_list])


        environment = Environment(loader=FileSystemLoader("templates/"))
        template = environment.get_template("prompt.txt")
        text_str = ". ".join(text_list)
        
        prev_posts = self.get_prev_posts(5)
        prev_post_count = len(prev_posts)
        
        today = datetime.today().strftime('%Y-%m-%d')
        prompt = template.render(text_list=text_str, caption_count= len(caption_list), captions=caption_str, prev_post_count=prev_post_count, prev_posts=prev_posts, date=today)
        #print(prompt)

        openai.api_key = self.config['open_ai_key']

        if "style" in conversation:
            style = conversation['style']
        else:
            style = get_random_style()
            bu.set_style(conversation['user_id'], style)

        environment = Environment(loader=FileSystemLoader("templates/"))
        template = environment.get_template("system_prompt.txt")
        system_prompt = template.render(style=style)


        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
        )

        contents = ''
        for choice in response.choices:
            contents += choice.message.content

        title = contents.split("\n")[0]
        contents = "\n".join(contents.split("\n")[1:])

        if title.startswith("Title:"):
            title = title.split("Title:")[1].strip()

        
        bu.set_status(conversation['user_id'], "Preview")
        bu.set_contents(conversation['user_id'], contents)
        bu.set_title(conversation['user_id'], title)

        contents = bu.get_contents(user_id)
        title = bu.get_title(user_id)
        preview = f"Title: {title}\n\n{contents}"
        self.bot.send_message(user_id,preview)


    def publish_task(self, user_id:str):
        '''Publishes the task to the blog'''
        conversation = bu.get_conversation(user_id)
        task_fn = bu.get_task_fn(user_id)

        re_comp = re.compile("\[Photo_[0-9]+\]",re.IGNORECASE)
        user = self.config['wordpress_user']
        password = self.config['wordpress_key']
        credentials = user + ':' + password
        token = base64.b64encode(credentials.encode())

        for m in conversation['messages']:
            if m['kind']=='media' and m.get('uploaded_href',None) is None:
                toUploadImagePath = task_fn+"/"+m['filename']
                mediaImageBytes = open(toUploadImagePath, 'rb').read()


                uploadImageFilename = m['filename']
                _, ext = os.path.splitext(m['filename'])
                curHeaders = {
                'Authorization': 'Basic ' + token.decode('utf-8'),
                "Content-Type": f"image/{ext[1:]}",
                "Accept": "application/json",
                'Content-Disposition': "attachment; filename=%s" % uploadImageFilename,
                }
                url = self.config['wordpress_url']
                if url.endswith("/"):
                    url = url[:-1]
                url = url+"/wp-json/wp/v2/media"

                resp = requests.post(
                url,
                headers=curHeaders,
                data=mediaImageBytes,
                )

                if resp.status_code != 201 and resp.status_code != 200:
                    raise Exception("Error uploading image: {}".format(resp.reason))

                jj = resp.json()

                if jj['mime_type'].startswith("video"):
                    m['uploaded_href'] = jj['source_url']
                    height = jj['media_details']['height']
                    width = jj['media_details']['width'] 
                    m['tag'] = f'[video width="{width}" height="{height}" mp4="{jj["source_url"]}"][/video]'
                else:
                    sizes = jj['media_details']['sizes']
                    for k in sizes:
                        width  = sizes[k]['width']
                        if width > 250 and width < 500:
                            m['uploaded_href'] = sizes[k]['source_url']
                            break
                    if m.get('uploaded_href',None) is None:
                        m['uploaded_href'] = jj['media_details']['sizes']['full']['source_url']

                    m['tag']=f"""<img src="{m['uploaded_href']}" alt="{m.get('text','')}" />"""

                bu.save_conversation(user_id, conversation)

        contents = conversation['contents']

        today = datetime.strftime(datetime.now(),"%Y/%m/%d")

        title = today + " - " + conversation['title']
        for m in conversation['messages']:
            if m['kind'] == 'media':
                if m.get('slug','') != '':
                    href = m['uploaded_href']
                    slug = m['slug']
                    caption = m.get("text","")
                    tag = m.get("tag","")
                    re_comp = re.compile(f"\[{slug}\]",re.IGNORECASE)
                    contents = re_comp.sub(f'<figure class="wp-block-image size-large">{tag}</figure><figcaption class="wp-caption-text">{caption}</figcaption>\n<br/>', contents)
                    # <figure class="wp-block-image size-large">[video width="720" height="1280" mp4="http://www.house4hack.co.za/wp-content/uploads/2023/06/d952c368-ff0b-45d6-8ba8-67071336142b.mp4"][/video]</figure>
                else:
                    href = m['uploaded_href']
                    contents += f'<figure class="wp-block-image size-large"><img src="{href}" alt=""/></figure>\n<br/>'

        contents += "\n\n<a href=\"https://www.house4hack.co.za/who-is-blogbot\">Who is Blogbot?</a>"


        user = self.config['wordpress_user']
        password = self.config['wordpress_key']
        credentials = user + ':' + password
        token = base64.b64encode(credentials.encode())
        header = {'Authorization': 'Basic ' + token.decode('utf-8')}

        post = {
        'title'    : title,
        'status'   : 'draft', 
        'content'  : contents,
        'categories': self.config['wordpress_category'], # category ID
        'date'   : datetime.now().isoformat().split('.')[0]
        }

        url = self.config['wordpress_url']
        if url.endswith("/"):
            url = url[:-1]
        responce = requests.post(url+"/wp-json/wp/v2/posts" , headers=header, json=post)
        if responce.status_code != 201 and responce.status_code != 200:
            raise Exception("Error publishing post: {}".format(responce.reason))
        
        jj = responce.json()
        #link = jj['link']
        id = jj['id']
        
        link = f"{url}/wp-admin/post.php?post={id}&action=edit"

        self.bot.send_message(user_id,f"Saved as draft on wordpress: {title}\n{link}")
        bu.set_status(user_id, "Published")

    def reload(self, user_id:str):
        '''Reloads the templates from git'''
        self.bot.send_message(user_id,"Reloading templates")


        result = subprocess.run(['git', 'pull'], stdout=subprocess.PIPE)
        reply = result.stdout.decode('utf-8')
        self.bot.send_message(user_id,"Reloaded:"+reply)
    


#if __name__=='__main__':
#    config = json.load(open("config.json"))
#    work_queue = queue.Queue()
#    worker = BlogProcessorWorker(work_queue, config)
#    worker.daemon = True
#    worker.start()
#
#    inFD = os.open('./fifo', os.O_RDWR | os.O_NONBLOCK)
#    sIn = os.fdopen(inFD, 'r')
#    while True:
#        select.select([sIn],[],[sIn])
#        for task in sIn:
#            try:
#                if task is not None:
#                    tasktype, user_id = task.strip().split(",")
#                    print(f"Starting {tasktype} for {user_id}")
#                    work_queue.put_nowait((tasktype, user_id))
#                    #work_queue.join()
#                time.sleep(1)
#                    
#
#            except Exception as e:
#                print(e)
#                traceback.print_exc(file=sys.stdout)

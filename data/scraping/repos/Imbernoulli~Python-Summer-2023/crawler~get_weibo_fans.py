import json
import random
import re
import threading
import urllib.parse
import urllib.request
from threading import Lock, Thread
from time import sleep

import jieba
import openai
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

def searchweibo(name):
    option = ChromeOptions()
    option.add_argument("--headless")
    browser = webdriver.Chrome()
    browser.get('https://weibo.com/login.php')
    with open('weibo_cookies.json', 'r', encoding='utf8') as f:
        listCookies = json.loads(f.read())
    for cookie in listCookies:
        cookie_dict = {
            'domain': '.weibo.com',
            'name': cookie['name'],
            'value': cookie['value'],
            "expires": '',
            'path': '/',
            'httpOnly': False,
            'HostOnly': False,
            'Secure': False
        }
        browser.add_cookie(cookie_dict)
    browser.get('https://weibo.com/login.php')
    sleep(1)
    WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "gn_search_v2")))
    searchinput = browser.find_element(By.CLASS_NAME, 'W_input')
    searchinput.send_keys(name)
    sleep(0.2)
    searchinput.send_keys(Keys.ENTER)
    new_window = browser.window_handles[-1]
    browser.switch_to.window(new_window)
    WebDriverWait(browser, 15).until(EC.presence_of_element_located((By.CLASS_NAME, "card-wrap")))
    weibo = browser.find_element(By.XPATH, '//div[@class="card card-user-b s-brt1 card-user-b-padding"]/div/a')
    weibo.click()
    # old_window = browser.window_handles[-2]
    browser.close()
    new_window = browser.window_handles[-1]
    browser.switch_to.window(new_window)
    current_url=browser.current_url
    browser.close()
    return current_url

def get_weibo_fans(url):
    print(url)
    option = ChromeOptions()
    option.add_argument("--headless")
    browser = webdriver.Chrome(options=option)
    browser.get('https://weibo.com/login.php')
    with open('weibo_cookies.json', 'r', encoding='utf8') as f:
        listCookies = json.loads(f.read())
    for cookie in listCookies:
        cookie_dict = {
            'domain': '.weibo.com',
            'name': cookie['name'],
            'value': cookie['value'],
            "expires": '',
            'path': '/',
            'httpOnly': False,
            'HostOnly': False,
            'Secure': False
        }
        browser.add_cookie(cookie_dict)
    sleep(1)
    browser.get(url)
    response = browser.page_source
    browser.close()
    fans = re.findall("粉丝<span>(.+?)</span>", response)[0]
    return fans

authors={}
pages=[]

def get():
    global pages
    for page in pages:
        if page["author"] in authors:
            continue
        else:
            if page["author_weibo_link"]!="":
                authorfans=page["author_fans"]
                authorlink=page["author_weibo_link"]
                authors[page["author"]]=page["author_fans"]
            else:
                try:
                    authorlink=searchweibo(page["author"])
                    try:
                        authorfans=get_weibo_fans(authorlink)
                    except Exception as e:
                        print(str(e))
                        authorfans=""
                    authors[page["author"]]=authorfans
                except Exception as e:
                    print(str(e))
                    authors[page["author"]]="0"
                    authorfans="0"
                    authorlink=""
            with open("author.txt", "a") as f:
                f.write(str(page["author"])+" "+str(authorfans)+" "+authorlink+"\n")

def main():
    global pages
    with open("refined_pages.json","r") as f:
        pages=json.loads(f.read())
    get()

if __name__ == "__main__":
    main()

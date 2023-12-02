import openai
import time
import requests
from selenium import webdriver
from selenium.webdriver import Chrome
import os
import json
import pandas as pd
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from tools.into_back import main as into_back
from tools.get_ac_name import main as get_ac_name
from tools.write_log import main as write_log
from tools.get_time_now import get_time

#登录
def login(cookie,params):
    name = cookie.strip(".txt").split("_")[1]
    if params['use_name']:
        #获取账号与用户名的字典
        _,dic = get_ac_name()
        name = dic[name]
    info = f'{name}: 登录成功'
    print(info)
    write_log(name,get_time()+' '+info,params)

    option = webdriver.ChromeOptions()
    if not params['display']:
        option.add_argument('headless')
    option.add_experimental_option("detach", True)
    option.add_experimental_option("excludeSwitches",["enable-logging"])
    url = 'https://cuttlefish.baidu.com/'
    web = Chrome(options=option)
    web.get(url)
    f = open(cookie)
    cookie_lst = json.load(f)
    for cookies in cookie_lst:
        web.add_cookie(cookies)
    web.refresh()
    info = f'{name}: 正在尝试进入后台'
    print(info)
    write_log(name,get_time()+' '+info,params)
    try:
        tips_icon = WebDriverWait(web, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="app"]/div[1]/div[1]/div/div[2]/a/div/i')))
        tips_icon.click()
    except:
        pass
    into_back(web,name,params)
    return web,name

def main(cookie,params):
    return login(cookie,params)

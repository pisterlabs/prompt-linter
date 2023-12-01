from selenium import webdriver
import requests
from bs4 import BeautifulSoup as bs
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import openai
import time
import random




def start():
    driver=webdriver.Chrome()
    driver.implicitly_wait(3)
    driver.get("https://web.whatsapp.com/")
    openai.api_key = "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"
    while True:
        try:
            qr_area=driver.find_element(By.XPATH,'//*[@id="app"]/div/div/div[3]/div[1]/div/div/div[2]/div')
            time.sleep(1)
        except:
            # wp_source=driver.page_source
            # soup=bs(wp_source,"html.parser")
            # grup=soup.find('div',{'class':''})
            # print(grup.text)
            # _8nE1Y
            time.sleep(3)
            persons=driver.find_elements(By.XPATH,"//*[@data-testid='cell-frame-title']")
            persons[2].click()
            time.sleep(1)
            
            # wp_source=driver.page_source
            # soup_end=bs(wp_source,"html.parser")
            # search=soup_end.find_all('span',{'class':['_11JPr','selectable-text', 'copyable-text']})

            # son_tweet=search[-1].find_all('span')[0].text
            
            
            message_list = driver.find_elements(By.XPATH,'.//div[@class="copyable-text"]')[-5:]
            for message in message_list:
                text = message.find_element(By.XPATH,'.//span[contains(@class, "selectable-text")]')
                print(text.text)
            # last_five_message=search[-5:]
            # for i in last_five_message:
            #     message_top=i.find_all('span')[0].text
            #     print(message_top)
            
            # messages=driver.find_elements(By.XPATH,"//*[@data-testid='msg-container']")
            # time.sleep(1)
            # last_five_message=messages[-5:]
            # for i in last_five_message:
            #     message_top=i.find_elements(By.XPATH,"//*[@class='_11JPr selectable-text copyable-text']")
            #     message_in=message_top[0].find_elements(By.XPATH,"span")
            #     print(message_in[0].text)
            print("you entered qr code")
            time.sleep(2)
    
start()
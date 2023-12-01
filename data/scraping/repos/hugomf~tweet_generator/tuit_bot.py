

import openai
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium import common
from selenium.webdriver.common.action_chains import ActionChains

import time

class TuitBot:
    
    def __init__(self, useProfile=True, user=None, password=None):
        #profile = FirefoxProfile("./profile")
        options = Options()
        #options.add_argument("-headless")
        if useProfile:
            options.profile = "./profile"
                
        driver = webdriver.Firefox(options=options)
        driver.get("https://twitter.com/home")

        self.bot = driver
        self.user = user
        self.password = password
        self.is_logged_in = False
        self.useProfile = useProfile

    def login(self):
        bot = self.bot
        bot.get('https://twitter.com/login')

        time.sleep(2)
        username_textbox = bot.find_element('xpath', '//input[@autocomplete="username"]')

        username_textbox.click()
        username_textbox.clear()
        username_textbox.send_keys(self.user)
        
        time.sleep(2)
        bot.find_element('xpath', '//div[@role="button"][2]').click()
        
        time.sleep(1)
        password = bot.find_element('xpath', '//input[@name="password"]')
        password.click()
        password.clear()
        password.send_keys(self.password)

        time.sleep(2)
        bot.find_element('xpath', '//div[@data-testid="LoginForm_Login_Button"]').click()


        time.sleep(4)
        self.is_logged_in = True

    def post(self, tweetBody, retries=3):
        
        bot = self.bot
        if not self.useProfile and not self.is_logged_in:
            raise Exception("You must log in first!")
    
        
        # try:
        #     bot.find_element(By.XPATH,"//a[@data-testid='SideNav_NewTweet_Button']").click()
        # except common.exceptions.NoSuchElementException:
        #     time.sleep(3)
        #     bot.find_element(By.XPATH,"//a[@data-testid='SideNav_NewTweet_Button']").click()


        print(f'el tweet fue: {tweetBody}')

        for i in range(retries):
            try:
                tweet_textbox = bot.find_element(By.XPATH,"//div[@data-testid='tweetTextarea_0']")
                time.sleep(2) 
                tweet_textbox.clear()
                time.sleep(1) 
                tweet_textbox.send_keys(tweetBody)
                time.sleep(2)
                tweet_textbox.send_keys(Keys.ARROW_UP)
                time.sleep(1)
                if ("#" in tweetBody):
                    tweet_textbox.send_keys(Keys.ESCAPE)
                    time.sleep(1)
                tweet_button = bot.find_element(By.XPATH,"//div[@data-testid='tweetButtonInline']")
                tweet_button.click()
                time.sleep(2)
                return "OK"
            except Exception as error:
                print('An exception occurred: {}'.format(error))
                try:
                    bot.find_element(By.XPATH,"//div[@data-testid='tweetTextarea_0']").send_keys(Keys.ESCAPE)
                except Exception as exc:
                    print('ESCAPE failed: {}'.format(exc))
                continue
            break
            time.sleep(2)  


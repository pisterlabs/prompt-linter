import os
import undetected_chromedriver as uc
import time
from selenium.webdriver.common.by import By
from openai import OpenAI

class PanoptoScraper:
    """
    Last Updated: 11/13/23 by Omer Mujawar
    
    A class to scrape captions from a video lecture in Panopto
    
    Attributes
    ----------
    credentials : dict
        Login credentials for HarvardKey to create, of the format { USERNAME: <username>, PASSWORD: <password> }
    url : string
        URL of Panopto lecture, of the format 'https://harvard.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=<video_id>'
    captions : list
        List of generated images.
    summary : string
        String 
        
    Methods
    -------
    scrape_lecture()
        Scrape lecture captions from the Panopto URL
    generate_summary()
        Generate summary of lecture from scraped lecture captions
    """

    def __init__(self, credentials, url):
        self.credentials = credentials
        self.url = url
    
    def scrape_lecture(self):
        # initialize headless Chrome driver
        chromeOptions = uc.ChromeOptions()
        chromeOptions.headless = True
        driver = uc.Chrome(use_subprocess=True, options=chromeOptions)

        URL = self.url
        USERNAME = self.credentials["username"]
        PASSWORD = self.credentials["password"]

        try:
            print("Beginning scraping...")
            
            # head to panopto page
            driver.get(URL)
            time.sleep(5)
            
            print("Located URL.")

            print("Clicking button to login")
            # click login button
            driver.find_element(By.ID, "PageContentPlaceholder_loginControl_externalLoginButton").click()
            time.sleep(10)

            print("Logging in")
            # HarvardKey login
            uname = driver.find_element(By.ID, "username")
            uname.send_keys(USERNAME)
            pw = driver.find_element(By.ID, "password")
            pw.send_keys(PASSWORD)

            driver.find_element(By.NAME, "submit").click()

            ## DUO process
            print("Waiting for a Duo push. Please check your phone!")
            time.sleep(15)

            ## After giving time to acknowledge DUO, click trust
            driver.find_element(By.ID, "trust-browser-button").click()
            time.sleep(8)

            try: 
                play_button = driver.find_element(By.ID, "playButton")
                print("Page loaded correctly!")
            except: 
                print("Error. Page did not load.")
                return 0

            captions = driver.find_elements(By.XPATH, "//li[contains(@id, 'UserCreatedTranscript')]/div[@class='index-event-row']/div[@class='event-text']/span")
            captions_list = []
            for caption in captions:
                captionText = caption.get_attribute("innerHTML")
                captions_list.append(captionText)
                # print(captionText)
            self.captions = captions_list

            print("Process completed.")
            return 1
        except:
            print("Error. Process not completed.")
            return 0

    def save_captions(self):
        if self.captions == []:
            print("Error. Captions not scraped. Please run scrape_lecture() first.")
            return 0
        
        print("Writing captions to captions.txt...")
        file = open("captions.txt", "w")
        file.writelines(self.captions)
        file.close()
        print("Completed writing. See captions.txt!")

    def generate_summary(self, OPENAI_API_KEY):
        if self.captions == []:
            print("Error. Captions not scraped. Please run scrape_lecture() first.")
            return 0
        
        captions_compiled = " ".join(self.captions)
    
        client = OpenAI(api_key = OPENAI_API_KEY)
        prompt = f"""You will read the following transcribed captions of a lecture. Please summarize this lecture into comprehensive academic notes.
        
        Captions: ${captions_compiled}

        Summary:"""
        response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
            "role": "user",
            "content": f"{prompt}"
            }
        ],
        temperature=1,
        max_tokens=800,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

        return response.choices[0].message.content

        

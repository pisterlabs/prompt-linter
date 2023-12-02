from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
from langchain.document_loaders.image import UnstructuredImageLoader
from django.db import models
from .VectorEngine import VectorEngine
from .Agent import Agent
import os



class KBItem(models.Model):
    '''
    Parent KBItem class to encapsulate data related to a "knowledge base item"
    URI - URI of website or document 
    userTags - User generated tags associated with the item
    UserID - UserID that owns this KBItem
    vectorEngine - object used to interact with the vector database 
    agent - Agent object to interact with LLM 
    '''
    
    URI = models.CharField(max_length = 500) 
    userTags = models.CharField()
    itemContent = models.TextField(default="")
    title = models.TextField(default="")
    sourceName = models.TextField(default="")
    userID = models.IntegerField()
    vectorEngine = VectorEngine()
    agent = Agent() 

        
    def parseURI(self): 
        '''
        Parent implementation of ParseURI instantiates the web driver to connect to a web URL. Will be overriden for specific
        scraping method and if the KBItem is not web based.
        '''
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        self.driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=chrome_options)
        self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'})

    def deployContentAgent(self): 
        '''
        Method to deploy the Agent to generate additional content related to the document
        '''
        self.userTags += self.agent.createItemTags(self.itemContent)
    
    def generateTitleAndSource(self):
        '''
        Method to generate title for document
        '''
        self.title,self.sourceName = self.agent.generateTitleAndSource(self.itemContent,self.URI)
    
    def addURI(self,URI, userTags = ""): 
        self.URI = URI; 
        self.userTags = userTags
        self.itemContent = ""


    
    def addUserTags(self,userTags):
        '''
        Set user tags method
        ''' 
        self.userTags = userTags
    
    
    def createVector(self, chunk_size = 100):  
        '''
        Method to call vectorEngine to convert text to embeddings and store in the vector database
        '''
        
        documents = self.vectorEngine.TextToDocs(text = self.itemContent,kbItemID=self.id,userID=self.userID)
        
        self.vectorEngine.storeVector(documents, chunk_size=chunk_size)



class ImageKBItem(KBItem): 
    
    def parseURI(self):
        super().parseURI()
        self.driver.set_window_size(1920, 1080)
        self.driver.get(self.URI)
        
        time.sleep(5)

        ssFile = "screen_shot" + str(self.id)+".png"
        self.driver.save_screenshot(ssFile)

        self.driver.close()
        self.driver.quit()

        loader = UnstructuredImageLoader(ssFile)
        data = loader.load() 

        os.remove(ssFile)

        for ele in data: 
            self.itemContent = self.itemContent + " " + ele.page_content   
    
    def deployContentAgent(self):
        self.itemContent = self.agent.trimItemContent(self.itemContent)
        super().deployContentAgent()

        
class TextKBItem(KBItem): 
    
    def parseURI(self): 
        super().parseURI()
        self.driver.get(self.URI)
        time.sleep(3)

        try: 
            alert = self.driver.switch_to.alert
            alert.dismiss() 
        except: 
            pass 

        # Get all 'p' and heading tags
        tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        for tag in tags:
            elements = self.driver.find_elements(By.TAG_NAME, tag)
            for element in elements:
                self.itemContent += element.text + "\n"

        print(self.itemContent)

        self.driver.quit()

#Command to remove embedding from table by kbItemID DELETE FROM langchain_pg_embedding WHERE cmetadata::jsonb @> '{"kbItemID": 37}'::jsonb;
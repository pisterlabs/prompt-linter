#Author: Yusuf Wadi
#Date Created 

# on the page with the context of the description and the users resume

# import libraries
from time import sleep
from selenium import common
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains as Actions
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA, load_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf2image.pdf2image import convert_from_path
# ====#
# from rake_nltk import Rake
import en_core_web_sm
import glob
import yaml
from langchain import OpenAI
import os
import requests
import base64
# ====#


class AutoApplier:
    # load self.config variables from self.config.yml
    def __init__(self, config: dict, profile: dict):

        self.config = self.verifyConfig(config)
        self.profile = self.verifyProfile(profile)
        #create autoApply folder in documents if it doesnt exist
        if not os.path.exists(os.path.normpath(os.path.expanduser("~/Documents/autoApply/"))):
            os.mkdir(os.path.normpath(os.path.expanduser("~/Documents/autoApply/")))
        # if unix
        if os.name == "posix":
            # path to resume in unix document folder
            self.resume = os.path.normpath( os.path.expanduser("~/Documents/autoApply/") + self.profile["resume"] + ".pdf")
        # if windows
        elif os.name == "nt":
            # path to resume in windows document folder
            self.resume = os.path.normpath(os.path.expanduser("~/Documents/autoApply/") + self.profile["resume"] + ".pdf")
        
        self.browser = None

    # ====#
    
    def verifyConfig(self, config: dict) -> dict:
        # verify config variables
        if config["driver_path"] is None:
            raise ValueError("driver_path is None")
        if config["query"] is None:
            raise ValueError("query is None")
        if config["batch_size"] is None:
            raise ValueError("batch_size is None")
        if config["scrolls"] is None:
            raise ValueError("scrolls is None")
        if config["inBatches"] is None:
            raise ValueError("inBatches is None")
        if config["role"] is None:
            raise ValueError("role is None")
        if config["llm"] is None:
            if config["model_path"] is None:
                raise ValueError("model_path is None, llm is None")
            raise ValueError("llm is None")
        if config["timeframe"] is None:
            raise ValueError("timeframe is None")
        return config
    
    def verifyProfile(self, profile: dict) -> dict:
        if profile["fname"] is None:
            raise ValueError("fname is None")
        if profile["lname"] is None:
            raise ValueError("lname is None")
        if profile["email"] is None:
            raise ValueError("email is None")
        if profile["phone"] is None:
            raise ValueError("phone is None")
        if profile["linkedin"] is None:
            raise ValueError("linkedin is None")
        if profile["website"] is None:
            raise ValueError("website is None")
        if profile["github"] is None:
            raise ValueError("github is None")
        if profile["visa"] is None:
            raise ValueError("visa is None")
        return profile
    
    def setup_browser(self):
        # define the path to the chromedriver executable
        driver_path = self.config["driver_path"]

        b_path = driver_path

        # define the path to the user profile
        userpp = self.config["user_profile"]
        options = webdriver.Chrome or webdriver.FirefoxOptions()
        options.binary_location = b_path

        # set dl options
        # prefs = {"download.default_directory": "C:/Users/thewa/Desktop/"}
        # e.g. C:\Users\You\AppData\Local\Google\Chrome\User Data

        userD = f"--user-data-dir={userpp}"
        options.add_argument(userD)
        options.add_argument(r'--profile-directory=Default')  # e.g. Profile 3
        # options.add_experimental_option("prefs", prefs)
        options.add_experimental_option("detach", True)
        self.browser = webdriver.Chrome or webdriver.Firefox(
            executable_path=ChromeDriverManager().install(), options=options)

    def setup_firefox(self):
        self.browser = webdriver.Firefox()

    def searchLinks(self, scrolls: int = 3, inBatches=False):
        # navigate to Google and search for the query
        self.close_all_tabs()
        query = self.config["query"] + " site:boards.greenhouse.io/*/jobs"
        self.browser.get("https://www.google.com/search?q=" + query + "&as_qdr=" + self.config["timeframe"])
        # search_box = self.browser.find_element(By.NAME, "q")
        # search_box.send_keys(query)
        # search_box.send_keys(Keys.RETURN)

        # scroll all the way down, wait for the search results to load, then scroll again
        for _ in range(scrolls):
            self.browser.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
            sleep(1)

        links = WebDriverWait(self.browser, 1).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, "a")))
        # 6 links per batch in parsed_links
        parsed_links = []
        # iterate over the links and open them
        for link in links:
            href = str(link.get_attribute("href"))
            if href.startswith("https://boards.greenhouse.io/"):
                parsed_links.append(href)

        # cut the parsed links into a list of batches
        if inBatches:
            batch_size = self.config["batch_size"]
            batches = [parsed_links[i:i + batch_size]
                       for i in range(0, len(parsed_links), batch_size)]

        if inBatches:
            return batches
        else:
            return parsed_links

    def linksFromLink(self, link: str):
        self.browser.get(link)
        sleep(1)
        links = WebDriverWait(self.browser, 1).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, "a")))
        # 6 links per batch in parsed_links
        parsed_links = []
        # iterate over the links and open them
        for link in links:
            href = str(link.get_attribute("href"))
            if href.startswith("https://boards.greenhouse.io/"):
                parsed_links.append(href)
        return parsed_links

    def openLinks(self, links: list):
        """
        opens a list of links given a webdriver
        returns None
        """
        self.browser._switch_to.window(self.browser.window_handles[0])
        for link in links:
            self.browser.execute_script(
                "window.open('" + link + "', '_blank');")

    def check_if_exists(self, id_):
        return len(self.browser.find_elements(By.ID, id_)) > 0
    
    def focusElement(self, element: webdriver.Chrome or webdriver.Firefox):
        """
        focuses an element given a webdriver
        returns None
        """
        self.browser.execute_script("arguments[0].scrollIntoView();", element)

    def selectDropGreen(self, field: webdriver.Firefox, value):
        """
        selects an option from a dropdown given a webdriver
        returns None
        specifically for greenhouse.io
        whoever made that site is sorry
        """
        value = f"\"{value}\""

        select = field.find_element(By.TAG_NAME, "select")
        select_id = select.get_attribute("id")
        vis_text = field.find_element(By.CLASS_NAME, "select2-chosen")
        vis_text_id = vis_text.get_attribute("id")
        try:
            self.browser.execute_script(
                f"document.getElementById(\"{select_id}\").value = {value};")

            # choose the option thats value attribute is equal to the value
            self.browser.execute_script(
                f"document.getElementById(\"{vis_text_id}\").textContent = Array.from(\
                                                                            document.getElementById(\"{select_id}\").\
                                                                                getElementsByTagName(\"option\")).\
                                                                                    filter(option => option.getAttribute(\"value\") == {value})[0]\
                                                                                        .textContent;")

            sleep(0.2)

        except common.exceptions.JavascriptException:
            print("JavascriptException")
            pass

    def simplify(self):
        sleep(0.5)
        application = self.browser.find_element(By.ID, "application_form")\
                if self.check_if_exists("application_form") else None
        main_fields = application.find_element(By.ID, "main_fields").find_elements(By.CLASS_NAME, "field")\
                if self.check_if_exists("main_fields") else None
        custom_fields = application.find_element(By.ID, "custom_fields").find_elements(By.CLASS_NAME, "field")\
                if self.check_if_exists("custom_fields") else None
        eeoc_fields = application.find_element(By.ID, "eeoc_fields").find_elements(By.CLASS_NAME, "field")\
                if self.check_if_exists("eeoc_fields") else None
        demographic_questions = application.find_element(By.ID, "demographic_questions").find_elements(By.CLASS_NAME, "field demographic_question ")\
                if self.check_if_exists("demographic_questions") else None
        
        if application:

            # main_fields
            if main_fields:
                for field in main_fields:
                    display_main = field.find_element(By.TAG_NAME, "label").text
                    self.focusElement(field)
                    match(display_main):
                        case _ if "First" in display_main:
                            # first name
                            field.find_element(By.TAG_NAME, "input").send_keys(
                                self.profile["fname"])
                        case _ if "Last" in display_main:
                            # last name
                            field.find_element(By.TAG_NAME, "input").send_keys(
                                self.profile["lname"])
                        case _ if "Email" in display_main:
                            # email
                            field.find_element(By.TAG_NAME, "input").send_keys(
                                self.profile["email"])
                        case _ if "Phone" in display_main:
                            # phone
                            field.find_element(By.TAG_NAME, "input").send_keys(
                                self.profile["phone"])
                        case _ if "Resume" in display_main:
                            # resume
                            sleep(1)
                            form = self.browser.find_element(
                                By.XPATH, '//*[@id="s3_upload_for_resume"]')
                            form.find_element(By.NAME, "file").send_keys(self.resume)
                            # form.submit()
                        case _ if "Cover" in display_main:
                            # cover letter
                            sleep(1)
                            cover_letter_path = os.path.normpath(
                                os.getcwd() + "/cover_letters/" + "temp" + ".txt")
                            form = self.browser.find_element(
                                By.XPATH, '//*[@id="s3_upload_for_cover_letter"]')
                            form.find_element(By.NAME, "file").send_keys(
                                cover_letter_path)
                            # form.submit()
                        case _ if "School" in display_main:
                            # school
                            field.find_element(By.TAG_NAME, "input").send_keys(self.profile["school"])
                        case _:
                            # raise(Warning("How did you get here"))
                            pass

            # custom_fields
            if custom_fields:
                for field in custom_fields:
                    # determine question type
                    question_type = field.find_element(By.TAG_NAME, "label").text
                    self.focusElement(field)
                    match(question_type):
                        case _ if "LinkedIn" in question_type:

                            field.find_elements(
                                By.TAG_NAME, "input")[-1].send_keys(self.profile["linkedin"])
                        case _ if "Website" in question_type:
                            field.find_elements(
                                By.TAG_NAME, "input")[-1].send_keys(self.profile["website"])
                        case [*_, "GitHub"]:
                            field.find_elements(
                                By.TAG_NAME, "input")[-1].send_keys(self.profile["github"])
                        case _ if "visa status" in question_type:
                            if self.profile["visa"]:
                                self.selectDropGreen(field, 1)
                            else:
                                self.selectDropGreen(field, 0)
                        case _:
                            # raise(Warning("How did you get here"))
                            pass
            # eeoc_fields
            if eeoc_fields:
                for field in eeoc_fields:
                    display_name = field.find_element(By.TAG_NAME, "label").text
                    self.focusElement(field)
                    match(display_name):
                        case _ if "Gender" in display_name:
                            # gender
                            if self.profile["gender"]:
                                self.selectDropGreen(field, 1)
                            else:
                                self.selectDropGreen(field, 2)
                            pass
                        case _ if "Hispanic" in display_name:
                            # hispanic or latino
                            if self.profile["hispanic"]:
                                self.selectDropGreen(field, "Yes")
                            else:
                                self.selectDropGreen(field, "No")

                        case _ if "Veteran" in display_name:
                            # veteran status
                            if self.profile["veteran"]:
                                self.selectDropGreen(field, 2)
                            else:

                                self.selectDropGreen(field, 1)
                        case _ if "Disability" in display_name:
                            # disability status
                            if self.profile["disabled"]:
                                self.selectDropGreen(field, 1)
                            else:
                                self.selectDropGreen(field, 2)
                        case _:
                            # raise(Warning("How did you get here"))
                            pass
            
            if demographic_questions:
                # all checkboxes
                for field in demographic_questions:
                    self.focusElement(field)
                    question = field.text
                    choices = field.find_elements(By.TAG_NAME, "label")
                    
                    match(question):
                        case _ if "ethnicity" in question:
                            #find self.profile[0] first if exists else self.profile[1]
                            print("ethnicity")  
                            for choice in choices:
                                if self.profile["race"][0] in choice.text:
                                    choice.find_element(By.TAG_NAME, "input").click()
                                    break
                                elif self.profile["race"][1] in choice.text:
                                    choice.find_element(By.TAG_NAME, "input").click()
                                    break
                        case _ if "gender" in question:
                            for choice in choices:
                                if self.profile["gender"]:
                                    if choice.text in ["Male", "Man"]:
                                        choice.find_element(By.TAG_NAME, "input").click()
                                        break
                                else:
                                    if choice.text in ["Female", "Woman"]:
                                        choice.find_element(By.TAG_NAME, "input").click()
                                        break
                        case _ if "veteran" in question:
                            for choice in choices:
                                if self.profile["veteran"]:
                                    if choice.text in ["Yes"]:
                                        choice.find_element(By.TAG_NAME, "input").click()
                                        break
                                else:
                                    if choice.text in ["No"]:
                                        choice.find_element(By.TAG_NAME, "input").click()
                                        break
                        case _ if "disability" in question:
                            for choice in choices:
                                if self.profile["disabled"]:
                                    if choice.text in ["Yes"]:
                                        choice.find_element(By.TAG_NAME, "input").click()
                                        break
                                else:
                                    if choice.text in ["No"]:
                                        choice.find_element(By.TAG_NAME, "input").click()
                                        break
                        case _:
                            # raise(Warning("How did you get here"))
                            pass
                
        else:
            print("No application form found")
            self.browser.close()

    def getJobDescKeys(self, job_desc: str):
        nlp = en_core_web_sm.load()
        keys = nlp(job_desc)
        job_desc_keys = str([token.text for token in keys.ents])
        job_desc_keys = job_desc_keys.strip("[],")

    def fillApps(self, links: list[list], model=None, inBatches: bool = False):
        # switch to the opened tabs
        try:  # not accounting search tab
            if inBatches:
                for idx, batch in enumerate(links):
                    self.openLinks(batch)
                    for window_handle in self.browser.window_handles[idx*len(batch):-1]:
                        self.doApp(window_handle, model=model)
                    next_apps = input(
                        "Press enter to continue to next batch of apps: ")
                    self.close_all_tabs(self.browser)
            else:  # default
                self.openLinks(links)
                for window_handle in self.browser.window_handles[1:]:
                    self.doApp(window_handle, model=model)
            # close all individual open tabs
            self.browser.execute_script("alert('All done!')")
            while True:
                # wait for user input q
                if input("Press q to quit: ") == "q":
                    self.close_all_tabs(self.browser)
                    break
        except KeyboardInterrupt:
            self.close_all_tabs(self.browser)

    def doApp(self, window_handle: str, model=None):
        self.browser.switch_to.window(window_handle)
        # wait for simplify popup to load
        sleep(1)
        # get company name from url
        company = self.browser.current_url.replace(
            "https://boards.greenhouse.io/", "").split("/")[0]
        # if the popup is present, press it
        try:
            # button is in shadow-root and has id fill-button
            # self.browser.execute_script(
            #     '''return document.querySelector("#simplifyJobsContainer > span").shadowRoot.querySelector("#fill-button").click()''')
            self.simplify()
            if model is not None:
                resume = self.load_pdf()
                self.llm_pass(model, company, resume)
            else:
                try:
                    submit = self.browser.find_element(By.ID, "submit_app")
                    self.focusElement(submit)
                    try:
                        submit.click()
                    except:
                        self.browser.execute_script(
                            "arguments[0].click();", submit)
                except common.NoSuchElementException:
                    print("No submit button found")
                    self.browser.close()
                    pass
                print(f"done with {company}")

        except common.exceptions.JavascriptException:
            # close tab, no popup
            self.browser.close()
            pass

    def llm_pass(self, model: GPT4All, company: str, resume):
        print("Using LLM")
        # find job description
        job_desc = self.browser.find_element(
            By.ID, "content").text
        print(job_desc)
        # find the key words in the job description
        job_desc_keys = self.getJobDescKeys(job_desc)

        print(job_desc_keys)

        # make cover letter
        if resume is None:
            print("No resume found")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=64
        )
        texts = text_splitter.split_documents(resume)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        db = Chroma.from_documents(
            texts, embeddings, persist_directory="db")
        qa = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            verbose=False,
        )
        ###

        cover_letter = self.create_cover(
            job_desc_keys=job_desc_keys, model=model, resume="", company=company, db=db, qa=qa)

        # find the custom questions

        print(f"done with {company}")

    def close_all_tabs(self,):
        for window_handle in self.browser.window_handles[1:]:
            self.browser.switch_to.window(window_handle)
            self.browser.close()
        self.browser.switch_to.window(self.browser.window_handles[0])

    def load_pdf(self):

        pdf_files = glob.glob(self.resume)

        if pdf_files:
            first_pdf = pdf_files[0]
            # do something with the first PDF file here
        else:
            print("No PDF files found in the resume directory.")
            return None
        loader = PyPDFLoader(first_pdf)
        document = loader.load_and_split()

        return document

    def create_cover(self, job_desc_keys: str, model: GPT4All, resume: str, company: str, db: Chroma, qa):

        cover_letter = None
        role = self.config["role"]
        prompt = f"Job Description Key Words: {job_desc_keys}\n\n Corressponding Cover Letter for {role} at {company} from applicant:\n\n"
        print(prompt)
        print("Generating cover letter...")
        if self.profile["local"]:
            cover_letter = qa(prompt)
            # write to file in cover_letters folder
            with open(f"cover_letters/{company}_cover_letter.txt", "w") as f:
                for line in cover_letter.split("\n"):
                    f.write(f"{line}\n")

        else:
            cover_letter = OpenAI().generate
            
            
            

        return cover_letter

    def activateLocalLM(self) -> GPT4All:
        # activate LLM from path
        # r"C:\Users\thewa\AppData\Local\nomic.ai\GPT4All4All\ggml-mpt-7b-instruct.bin"
        model = GPT4All(model=self.config["model_path"], n_ctx=1024, n_batch=1, n_threads=8,
                        n_parts=1, n_predict=1, seed=42, f16_kv=False, logits_all=False, vocab_only=False, use_mlock=False, embedding=False)
        return model if isinstance(model, GPT4All) else None

    def main(self, model=None):
        if model is not None:
            print("LLM is active.")
            print(model)
        scrolls = int(self.config["scrolls"])
        inBatches = self.config["inBatches"]
        self.browser = self.setup_self.browser()
        links = self.searchLinks(scrolls=scrolls)
        self.fillApps(links, model=model if isinstance(
            model, GPT4All) else None, inBatches=inBatches)

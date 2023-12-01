"""Welcome to Reflex! This file outlines the steps to create a basic app."""
from rxconfig import config
from typing import List
import reflex as rx

import os, glob, json, shutil
from time import sleep
import subprocess
import re
import pandas as pd
from io import StringIO
import requests
import io

import random
from PIL import Image
from newspaper import Article
import urllib.request

import openai
openai.organization = "org-MGwFb1CrjeNZupHeaarBRyrN"
openai.api_key = os.getenv("OPENAI_API_KEY") or "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"
ram_url = "http://172.31.27.12:3000"
# from ram.models import tag2text_caption
# from ram import inference_tag2text as inference
# from ram import get_transform
# pretrained = '/home/dt/Projects/Work/citrusberry/recognize-anything/tag2text_swin_14m.pth'
# image_size = 384
# thre = 0.68
# specified_tags = 'None'
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'
# transform = get_transform(image_size=image_size)
# delete_tag_index = [127,2961, 3351, 3265, 3338, 3355, 3359]
# # @st.cache_data  # 👈 Add the caching decorator
# # @rx.memo


# def load_model():
#     model = tag2text_caption(pretrained=pretrained,
#                                 image_size=image_size,
#                                 vit='swin_b',
#                                 delete_tag_index=delete_tag_index)
#     model.threshold = thre  # threshold for tagging
#     model.eval()
#     model = model.to(device)
#     return model

# model = load_model()


filename = f"{config.app_name}/{config.app_name}.py"
colindex = 0
accent_color = "#a883ed"
style = {

    "background" : "rgb(250, 250, 250)",
    # "font_family": "AirCerealMedium",
    # "font_family": "'Ariel', sans-serif",
    "font_size": "16px",
    # "font_weight": "",    
    "::selection": {
            "background_color": accent_color,
        },  
    "th" : {
            "background":"linear-gradient(45deg,#e6e4fc,#fceded)",  
    },
#Disco button


#Disco Button


    rx.ResponsiveGrid: {
        "animation": "fadeInAnimation ease 3s",
        "animation-iteration-count": "1",
        "animation-fill-mode": "forwards",
    },

    rx.Heading: {
        # "font_size": "32px",
        "font_family": "AirCereal",
        # "font_family": "'Ariel', sans-serif",
        "font_weight": "700",
        "color": "#a61d55",
    },
    rx.Text: {
        "font_family": "AirCerealNormalText",
        "line-height" : "1.7",
        # "font_weight": "100",
        "font_size": "16px",
        "font-weight": "normal",
        # "font-variant": "normal"
    },
    rx.Card: {
        "border-radius" : "16px",
        # "box-shadow" : "5px 10px",
        # "box-shadow" : "rgb(204, 219, 232) 3px 3px 6px 0px inset, rgba(255, 255, 255, 0.5) -3px -3px 6px 1px inset;"
        "box-shadow" : "6px 6px 12px #b8b9be,-6px -6px 12px #fff!important",
        "padding" : "10px 20px",
        "margin" : "10px 20px",
        # "background" : ""
    },
    rx.Badge: {
        "padding" : "10px 20px!important",
        "margin" : "10px 20px!important",   
        "text-transform" : "lowercase!important",
        "border-radius" : "5px!important",
        "box-shadow" : "5px 5px #000000!important",
    },
    rx.Slider: {
        "height": "5px",
        "overflow": "hidden",
        "background": "#fff",
        # "border" : "1px solid #29d",
    },

    rx.DataTable: {
        "background": "linear-gradient(45deg,#e6e4fc,#fceded)",
    }
    # rx.SliderFilledTrack: {
	# "position": "absolute",
	# "top": "0",
	# "right": "100%",
	# "height": "5px",
	# "width": "100%",
	# "background": "#29d",
    # }    

    rx.Tabs: {
        "background" : "linear-gradient(45deg,#e6e4fc,#fceded) !important",
        "color" : "#a61d55",
        "shadow" : "lg",
        "border-radius" : "16px",
        "box-shadow" : "6px 6px 12px #b8b9be,-6px -6px 12px #fff!important",

    }


}


class ArticleData:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.data_file = os.path.join(self.data_dir, 'articles.json')
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w') as f:
                json.dump([], f)

    def store_article(self, article):
        """
        Store article data to json file
        """
        with open(self.data_file, 'r+') as f:
            articles = json.load(f)
            articles.append(article)
            f.seek(0)           # reset file position to the beginning.
            json.dump(articles, f, indent=4)

    def get_articles(self):
        """
        Fetch all articles from json file
        """
        with open(self.data_file, 'r') as f:
            articles = json.load(f)
            return articles
    def delete_articles(self,article_url):
        """
        Delete a specific article from json file
        """        
        with open(self.data_file, 'r+') as f:
            articles = json.load(f)
            articles = [article for article in articles if article['url'] != article_url]
            f.seek(0)           # reset file position to the beginning.
            f.truncate()        # remove existing file content.
            json.dump(articles, f, indent=4)        



article_data = ArticleData()
articles = article_data.get_articles()
a_options : List[str] = [datavalue['url'] for datavalue in articles]

b_options = []

def fetch_preloaded(self,input_url,article):
    screendata = article
    global model
    global articles
    self.processing,self.complete = False,False
    yield
    sleep(2)
    import pdb;pdb. set_trace()
      




class State(rx.State):
    # The colors to cycle through.
    global a_options
    colors: List[str] = [
        "black",
        "red",
        "green",
        "blue",
        "purple",
    ]
    # selected_option_a: str = "No selection yet."
    # selected_option_a: str = a_options[0]
    selected_option_a: str = ""
    text: str = "Enter Page URL"
    #text: str =  a_options[0]
    processing = False
    complete = False
    error_occured = False
    image_url = ""
    model_tags = ""
    model_caption = ""
    alert_msg = ""
    alert_show: bool = False
    alert_msg_header = ""
    article_text = ""
    article_title = ""
    model_tag_list: List[str] = ['1','2']
    # The index of the current color.
    index: int = 0
    df = pd.DataFrame([[]])
    df1 = pd.DataFrame([[]])
    pff1 = ""
    pfl1 = ""
    df2 = pd.DataFrame([[]])
    pff2 = ""
    pfl2 = ""
    df3 = pd.DataFrame([[]])
    pff3 = ""
    pfl3 = ""
    df4 = pd.DataFrame([[]])
    pff4 = ""
    pfl4 = ""
    df5 = pd.DataFrame([[]])
    pff5 = ""
    pfl5 = ""
    df6 = pd.DataFrame([[]])
    pff6 = ""
    pfl6 = ""

    #Openai model outputs
    iab_safety_response_msg = ""
    iab_response_msg = ""
    global_brands_response_msg = ""
    indian_brands_response_msg = ""
    web_inventory_response_msg = ""
    ind_web_response_msg = ""
    news_response_msg = ""
    news_ind_response_msg = ""
    sentiment_filled_bg = "red"
    sentiment_empty_bg = "green.100"
    sentiment_color = sentiment_empty_bg.split('.')[0]
    sentiment_value = 0
    sff1 = ""
    sentiment_disp_value = 0
    keyword_list = pd.DataFrame([[]])
    # keyword_list: List[str] = [["1"]]
    

    def next_color(self):
        """Cycle to the next color."""
        self.index = (self.index + 1) % len(self.colors)

    @rx.var
    def color(self) -> str:
        return self.colors[self.index]
    
    def clear_text(self):
        # import pdb;pdb.set_trace()
        self.text = ""    



    def run_analysis(self):
        screendata = {}
        global model
        global articles
        self.processing,self.complete = False,False
        yield        
        if self.text == "":
            self.text = self.selected_option_a
            if self.text == "Select an example." or self.text == "No selection yet." or self.text == "":
                self.alert_msg_header = "Error"
                self.alert_msg = "Please enter url link or select a link from the dropdown"
                self.alert_change()
                self.processing,self.complete = False,True            
                yield
                return 
        self.processing,self.complete = True,False
        self.iab_safety_response_msg = ""
        self.iab_safety_response_msg = ""
        self.iab_response_msg = ""
        self.global_brands_response_msg = ""
        self.indian_brands_response_msg = ""
        self.web_inventory_response_msg = ""
        self.ind_web_response_msg = ""
        self.news_response_msg = ""
        self.news_ind_response_msg = ""
        self.sentiment_filled_bg = "red"
        self.sentiment_empty_bg = "green.100"
        self.sentiment_value = 0
        self.sff1 = ""
        self.sentiment_disp_value = 0   
        self.pff1 = " "
        self.pfl1 = " "
        self.pff2 = " "
        self.pfl2 = " "
        self.pff3 = " "
        self.pfl3 = " "
        self.pff4 = " "
        self.pfl4 = " "
        self.pff5 = " "
        self.pfl5 = " "
        self.pff6 = " "
        self.pfl5 = " "
        self.pfl6 = " "           
        yield
        for article in articles:
            if article['url'] == self.text:
                screendata = article       
                article_image = screendata["image"]
                imgpost = article_image.split('.')[-1]
                article_title = screendata["title"]
                print(f"Article image file is - {article_image}")
                sleep(1)
                try:
                    os.system(f"wget {article_image} -O downloads/mainimage.{imgpost}")
                    list_uploaded_file = f"downloads/mainimage.{imgpost}"
                    uploaded_file = list_uploaded_file    
                    ui_uploaded_file = f'mainimage.{imgpost}'    
                    shutil.move(uploaded_file,f'assets/{ui_uploaded_file}')
                    uploaded_file = "assets/"+ui_uploaded_file
                except IndexError as error:
                    print("Image file doesnt exist")
                    uploaded_file = "doesntexist.txt"  
                except Exception as error:  
                    print(error)
                    uploaded_file = "doesntexist.txt"  
                if article_title and os.path.exists(uploaded_file):
                    print("Main Execution")
                    self.processing,self.complete = False,True
                    image = Image.open(uploaded_file)
                    self.image_url = image 
                    self.article_title = article_title
                    self.article_text = screendata["text"]
                    self.model_tag_list =  screendata["tags"]
                    self.model_caption = screendata["caption"]
                    try: 
                        self.sff1 = screendata["sff1"]

                        self.keyword_list = pd.DataFrame(screendata["keyword_list"])

                        self.sentiment_disp_value = screendata["sentiment_disp_value"]

                        self.sentiment_value = int(self.sentiment_disp_value * 10)
                        self.sentiment_disp_value = f"{self.sentiment_value}%"
                        self.sentiment_filled_bg = screendata["sentiment_filled_bg"]
                        self.sentiment_empty_bg = screendata["sentiment_empty_bg"]   
                        self.sentiment_color = screendata["sentiment_empty_bg"].split('.')[0]
                    except Exception as error:
                        print(error)
                        try:
                            self.iab_safety_response_msg = screendata["sff1"]
                        except:
                            print(error)
                    yield
                    print("Get IAB Categories")
                    try:
                        self.df = pd.DataFrame(screendata["df"])  
                    except Exception as error:
                        print(error)
                        self.error_occured = True
                        self.iab_response_msg = screendata["df"]
                    yield
                    print("Get Brands")
                    try:
                        self.pff1 = screendata["pff1"]
                        self.df1 = pd.DataFrame(screendata["df1"]) 
                        self.pfl1 = screendata["pfl1"]
                    except Exception as error:
                        print(error)
                        self.pff1 = screendata["pff1"]
                    yield    
                    print("Indian Brands")
                    try:
                        self.pff2 = screendata["pff2"]
                        self.df2 = pd.DataFrame(screendata["df2"]) 
                        self.pfl2 = screendata["pfl2"]
                    except Exception as error:
                        print(error)
                        self.pff2 = screendata["pff2"]
                    yield
                    print("Websites")
                    try:
                        self.pff3 = screendata["pff3"]
                        self.df3 = pd.DataFrame(screendata["df3"]) 
                        self.pfl3 = screendata["pfl3"]
                    except Exception as error:
                        print(error)
                        self.pff3 = screendata["pff3"]
                    yield  

                    print("Indian Websites")
                    try:
                        self.pff4 = screendata["pff4"]
                        self.df4 = pd.DataFrame(screendata["df4"]) 
                        self.pfl4 = screendata["pfl4"]
                    except Exception as error:
                        print(error)
                        self.pff4 = screendata["pff4"]
                    yield
                    print("News")
                    try:
                        self.pff5 = screendata["pff5"]
                        self.df5 = pd.DataFrame(screendata["df5"]) 
                        self.pfl5 = screendata["pfl5"]
                    except Exception as error:
                        print(error)
                        self.pff5 = screendata["pfl5"]
                    yield
                    print("News India")        
                    try:
                        self.pff6 = screendata["pff6"]
                        self.df6 = pd.DataFrame(screendata["df6"]) 
                        self.pfl6 = screendata["pfl6"]        
                    except Exception as error:
                        print(error)
                        self.pff6 = screendata["pff6"]
                    yield
                    return True               
                elif uploaded_file == "parisingerror":
                    self.alert_msg_header = "Error"
                    self.alert_msg = "Failed to parse url"
                    self.alert_change()
                    self.processing,self.complete = False,True
                    yield                
                elif not os.path.exists(uploaded_file):
                    self.alert_msg_header = "Error"
                    self.alert_msg = "Failed to load image"
                    self.alert_change()
                    self.processing,self.complete = False,True
                    yield
                elif not article.title:
                    self.alert_msg_header = "Error"
                    self.alert_msg = "Failed to load data file"
                    self.alert_change()
                    self.processing,self.complete = False,True
                    yield            
                else:
                    self.alert_msg_header = "Error"
                    self.alert_msg = "Unknown Error"
                    self.alert_change()
                    self.processing,self.complete = False,True
                    yield                                   
                return True
        removestatus = [os.remove(file) for file in glob.glob('downloads/*')]
        removestatus = [os.remove(file) for file in glob.glob('assets/mainimage*')]
        article = Article(self.text)
        article.download()
        # article.html
        article.parse()
        article_image = article.top_image        
        imgpost = article_image.split('.')[-1]
        print(f"Article image file is - {article_image}")
        try:
            # Bug 001 - Websites like khaleej times mask image url, download through wget fails
            #os.system(f"wget {article_image} -O downloads/mainimage.{imgpost}")
            if 'jpg' in article_image or 'png' in article_image or 'jpeg' in article_image:
                os.system(f"wget {article_image} -O downloads/mainimage.{imgpost}")
            else:
                response = requests.get(article_image)
                imgpost = 'jpg'
                # Check if the request was successful
                if response.status_code == 200:
                    # Write the content of the response to a file
                    with open(f'downloads/mainimage.{imgpost}', 'wb') as file:
                        file.write(response.content)                            
            list_uploaded_file = f"downloads/mainimage.{imgpost}"
            uploaded_file = list_uploaded_file    
            ui_uploaded_file = f'mainimage.{imgpost}'    
            shutil.move(uploaded_file,f'assets/{ui_uploaded_file}')
            uploaded_file = "assets/"+ui_uploaded_file      
        except IndexError as error:
            print("Image file doesnt exist")
            uploaded_file = "doesntexist.txt"  
        except Exception as error:  
            print(error)
            uploaded_file = "doesntexist.txt"  
        if article.title and os.path.exists(uploaded_file):
            print("Main Execution")
            # image = Image.open(uploaded_file)
            
            # self.image_url = ui_uploaded_file 
            print(uploaded_file)
            image = Image.open(uploaded_file)
            self.image_url = image 
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            # image = transform(image).unsqueeze(0).to(device)   
            byte_arr = io.BytesIO()
            image.save(byte_arr, format='JPEG')
            byte_arr = byte_arr.getvalue()            

            # POST request to the Flask API
            url = f"{ram_url}/analyze_image"
            try:
                response = requests.post(
                    url,
                    files={
                        'image': ('image.jpg', byte_arr, 'image/jpeg')
                    }
                    , timeout=10
                )
            except Exception as connerror:
                print(connerror)
                self.alert_msg_header = "Error"
                self.alert_msg = "Image Processing Demo Service is Down - Will be up from 9:00 AM to 8:30 AM IST"
                self.alert_change()
                self.processing,self.complete = False,False
                yield 
                return True                  
            # Print the response
            print(response.json())      
            res = response.json()       
            # res = inference(image, model, specified_tags)
            self.article_title = article.title
            if 'Register to read and get full access to gulfnews.com' in article.text:
                self.article_text = f'This is a premium article. The text is unavailable - {article.text[0:2000]}'
            else:
                self.article_text = article.text[0:2000]
            print("Model Identified Tags: ", res[0])
            print("User Specified Tags: ", res[1])
            print("Image Caption: ", res[2])     
            self.model_tag_list =  res[0].split('|')  
            self.model_caption = res[2]
            self.processing,self.complete = False,True
            yield            
            # for i in range(0,10):
            try: 
                mesg1 = {"role":"user","content":f"Please analyze the sentiment of the following message, and get the top 10 keywords or keyphrases - {self.article_text}. Rate the sentiment on a scale of 1 to 10 with 10 being extremely positive and 1 being extremely negative. Send response as first paragraph with only one word describing the sentiment of the article - positive.,negative. or neutral. ,preceded by 'Sentiment:' and followed by two next line characters, list of only 10 keywords separated by next line and last paragraph giving the rating.Show only the numeric value of the rating in curly braces."}
                # mesg1 = {"role":"user","content":f"Please analyze the sentiment of the following message, and get the top 10 keywords or keyphrases - {self.article_text}. Rate the sentiment on a scale of 1 to 10 with 10 being extremely positive and 1 being extremely negative. Send response as first paragraph with only one word describing the sentiment of the article - positive.,negative. or neutral. ,preceded by 'Sentiment:', list of only 10 keywords separated by next line and last paragraph giving the rating.Show only the numeric value of the rating in curly braces."}
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role":"system","content":f"Please respond as a content expert and consultant and do not include disclaimers."},
                        mesg1
                        # {"role":"user","content":f"Suggest brands whose tone of voice resonates with following tags - {res[0]} - {res[2]}"}
                    ],
                    request_timeout=20
                )
                iab_safety_response_msg = response["choices"][0]["message"]
                text = iab_safety_response_msg["content"]
                print(text)
                paragraphs = text.split('\n\n')
                print(paragraphs)
                try:
                    self.sff1 = paragraphs[0].split('Sentiment:')[-1].split('.')[0].strip() + " - "
                except:
                    self.sff1 = paragraphs[0].split('Sentiment:')[-1].strip() + " - "
                # import pdb;pdb.set_trace()

                table_content = paragraphs[1]
                brand_names = table_content.replace('`','').replace('|','').split('\n')
                data = []
                for match in brand_names:
                    if 'Key' in match or 'key' in match or '---' in match or not match or 'rand' in match:
                        continue
                    # image_tag = match[0]
                    # iab_category = match[1]
                    keyword = match
                    data.append({'Keywords': keyword})
                    if len(data) > 10:
                        break
                self.keyword_list = pd.DataFrame(data)

                # self.iab_safety_response_msg = iab_safety_response_msg["content"]
                para2 = paragraphs[-1] if match not in paragraphs[-1] else ""
                match = re.search(r"{([0-9.]+)}$", para2)
                if match:
                    self.sentiment_disp_value = float(match.group(1))  

                    self.sentiment_value = int(self.sentiment_disp_value * 10)
                    self.sentiment_disp_value = f"{self.sentiment_value}%"
                else:
                    self.sentiment_value = 0
                if self.sentiment_value <= 40:
                    self.sentiment_filled_bg = "red"
                    self.sentiment_empty_bg = "red.100"
                    self.sentiment_color = "red"
                elif self.sentiment_value >= 60:
                    self.sentiment_filled_bg = "green"
                    self.sentiment_empty_bg = "green.100"
                    self.sentiment_color = "green"
                else:
                    self.sentiment_filled_bg = "grey"
                    self.sentiment_empty_bg = "grey.100"
                    self.sentiment_color = "grey"
            except Exception as error:
                print(error)
                try:
                    self.iab_safety_response_msg = text
                except:
                    print(error)
                    
                    


            yield
            # st.info(f'X-rae Response -  {iab_response_msg["content"]}')
            print("Get IAB Categories")
            sysmessage = {"role":"system","content":f"Please respond as a content expert and consultant and do not include disclaimers.Please provide information in tabular format or in a format which can be parsed into a table using a python API."}
            message1 = {"role":"user","content":f"Please take a guess of the IAB categories for an article with following image tags and caption - {res[0]} - {res[2]}. Return a table with single column of IAB category list "}
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        sysmessage,
                        message1,
                        # {"role":"user","content":f"Suggest brands whose tone of voice resonates with following tags - {res[0]} - {res[2]}"}
                    ],
                    request_timeout=20
                )
                iab_response_msg = response["choices"][0]["message"]
                # Convert response to data
                text = iab_response_msg["content"]
                # matches = re.findall(r'(\w+)\s*->\s*(.*)', text)
                matches = re.findall(r'\| (.*) \|\n', text)
                data = []
                for match in matches:
                    if 'IAB' in match:
                        continue
                    # image_tag = match[0]
                    # iab_category = match[1]
                    iab_category = match
                    # data.append({'Image Tag': image_tag, 'IAB Category': iab_category})
                    data.append({'IAB Category': iab_category})
                # Create a DataFrame from the data
                # print(data)
                self.df = pd.DataFrame(data)                        
            except Exception as error:
                print(error)
                self.error_occured = True
                self.iab_response_msg = text
            # self.iab_response_msg = iab_response_msg["content"]
            yield
            print("Get Brands")
            # import pdb;pdb.set_trace()
            reply1 = {"role":"assistant","content":f"{iab_response_msg['content']}"}
            # for trial in range(0,10):
            message2 = {"role":"user","content":f"Suggest list of top 10 global brands whose tone of voice resonates with following tags - {res[0]} - {res[2]}.Return response as a table with single column of Brand names without index"}
            #Return a table as comma-separated values with single column of Brand names
            # # st.info(f'X-rae Response -  {iab_response_msg["content"]}')
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            sysmessage,
                            message1,
                            reply1,
                            message2
                    ],
                    request_timeout=20
                )          
                global_brands_response_msg = response["choices"][0]["message"]
                text = global_brands_response_msg["content"]
                print(text)
                paragraphs = text.split('\n\n')
                # Extract last paragraph
                # last_paragraph = re.sub(paragraph_pattern, '', text, count=1, flags=re.MULTILINE | re.DOTALL)

                # Extract the first paragraph
                self.pff1 = paragraphs[0]

                # Extract the table content
                table_content = paragraphs[1]
                brand_names = table_content.replace('`','').replace('|','').split('\n')
                data = []
                for match in brand_names:
                    if 'IAB' in match or '---' in match or not match or 'rand' in match:
                        continue
                    # image_tag = match[0]
                    # iab_category = match[1]
                    iab_category = match
                    # data.append({'Image Tag': image_tag, 'IAB Category': iab_category})
                    data.append({'Brand Names': iab_category})
                # Create a DataFrame from the data
                # print(data)
                self.df1 = pd.DataFrame(data) 
                # Create a DataFrame from the table content
                # self.df1 = pd.read_csv(StringIO(table_content), skipinitialspace=True)

                # Extract the last paragraph
                self.pfl1 = paragraphs[-1] if match not in paragraphs[-1] else ""
                # self.global_brands_response_msg = global_brands_response_msg["content"]
            except Exception as error:
                print(error)
                self.pff1 = text
            yield    
            print("Indian Brands")
            reply2 = {"role":"assistant","content":f"{global_brands_response_msg['content']}"}
            message3 = {"role":"user","content":f"Suggest list of top 10 Indian brands whose tone of voice resonates with following tags - {res[0]} - {res[2]}. Return response as a table with single column of Brand names without index"}
            # # st.info(f'X-rae Response -  {global_brands_response_msg["content"]}')
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            sysmessage,
                            message1,
                            reply1,
                            message2,
                            reply2,
                            message3,
                    ],
                    request_timeout=20
                )          
                indian_brands_response_msg = response["choices"][0]["message"]
                # self.indian_brands_response_msg = indian_brands_response_msg["content"]
                text = indian_brands_response_msg["content"]
                print(text)
                paragraphs = text.split('\n\n')
                # Extract the first paragraph
                self.pff2 = paragraphs[0]

                # Extract the table content
                table_content = paragraphs[1]

                brand_names = table_content.replace('`','').replace('|','').split('\n')
                data = []
                for match in brand_names:
                    if 'IAB' in match or '---' in match or not match or 'rand' in match:
                        continue
                    # image_tag = match[0]
                    # iab_category = match[1]
                    iab_category = match
                    # data.append({'Image Tag': image_tag, 'IAB Category': iab_category})
                    data.append({'Brand Names': iab_category})
                self.df2 = pd.DataFrame(data) 

                # # Create a DataFrame from the table content
                # self.df2 = pd.read_csv(StringIO(table_content), skipinitialspace=True)

                # Extract the last paragraph
                self.pfl2 = paragraphs[-1] if match not in paragraphs[-1] else ""
            except Exception as error:
                print(error)
                self.pff2 = text
            yield
            print("Websites")
            reply3 = {"role":"assistant","content":f"{indian_brands_response_msg['content']}"}
            message4 = {"role":"user","content":f"Suggest the right list of top 10 global website inventory to run the aboce brand ads along with the IAB categories. Return a table as comma-separated values with two columns - IAB Category and Website Name"}
            # # st.info(f'X-rae Response -  {indian_brands_response_msg["content"]}')    
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            sysmessage,
                            message1,
                            reply1,
                            message2,
                            reply2,
                            message3,
                            reply3,
                            message4,
                    ],
                    request_timeout=20
                )          
                web_inventory_response_msg = response["choices"][0]["message"]
                # self.web_inventory_response_msg = web_inventory_response_msg["content"]
                text = web_inventory_response_msg["content"]
                if "apologize" in text or 'unfortunately' in text or 'Unfortunately' in text or "Apologize" in text or "Apologise" in text or "apologise" in text:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                                sysmessage,
                                message1,
                                reply1,
                                message2,
                                reply2,
                                message3,
                                reply3,
                                message4,
                        ],
                        request_timeout=20
                    )          
                    web_inventory_response_msg = response["choices"][0]["message"]
                    # self.web_inventory_response_msg = web_inventory_response_msg["content"]
                    text = web_inventory_response_msg["content"]
                print(text)
                paragraphs = text.split('\n\n')
                # Extract the first paragraph
                self.pff3 = paragraphs[0]

                # Extract the table content
                table_content = paragraphs[1]
                matches = table_content.split('\n')
                data = []
                for match in matches:
                    if 'IAB' in match or '---' in match or not match or 'rand' in match:
                        continue
                    iab_category,webname = match.split(',')
                    data.append({'Website Name': webname, 'IAB Category': iab_category})                                
                self.df3 = pd.DataFrame(data) 

                # Create a DataFrame from the table content
                # self.df3 = pd.read_csv(StringIO(table_content), skipinitialspace=True)
                # self.df3 = pd.DataFrame([row.split(':') for row in table_content], columns=['Category', 'Websites'])

                # Strip leading and trailing whitespace from the DataFrame
                # self.df3 = self.df3.apply(lambda x: x.str.strip())            

                # Extract the last paragraph
                self.pfl3 = paragraphs[-1] if match not in paragraphs[-1] else ""
            except Exception as error:
                print(error)
                self.pff3 = text
            yield                 
            print("Indian Websites")
            reply4 = {"role":"assistant","content":f"{web_inventory_response_msg['content']}"}
            message5 =  {"role":"user","content":f"Suggest the right list of top 10 Indian website inventory to run the following brand ads along with the IAB categories. Return a table as comma-separated values with two columns - IAB Category and Website Name"}   
            # # st.info(f'X-rae Response -  {web_inventory_response_msg["content"]}')    
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            sysmessage,
                            message1,
                            reply1,
                            message2,
                            reply2,
                            message3,
                            reply3,
                            message4,
                            reply4,
                            message5,
                    ],
                    request_timeout=20
                )                 
                # import pdb;pdb.set_trace()
                ind_web_response_msg = response["choices"][0]["message"]
                # self.ind_web_response_msg = ind_web_response_msg["content"]
                text = ind_web_response_msg["content"]
                print(text)
                if "apologize" in text or 'unfortunately' in text or 'Unfortunately' in text or "Apologize" in text or "Apologise" in text or "apologise" in text:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                                sysmessage,
                                message1,
                                reply1,
                                message2,
                                reply2,
                                message3,
                                reply3,
                                message4,
                                reply4,
                                message5,
                        ],
                        request_timeout=20
                    )          
                    # import pdb;pdb.set_trace()
                    ind_web_response_msg = response["choices"][0]["message"]
                    # self.ind_web_response_msg = ind_web_response_msg["content"]
                    text = ind_web_response_msg["content"]
                    print(text)
                paragraphs = text.split('\n\n')
                # Extract the first paragraph
                self.pff4 = paragraphs[0]

                # Extract the table content
                table_content = paragraphs[1]
                matches = table_content.split('\n')
                data = []
                for match in matches:
                    if 'IAB' in match or '---' in match or not match or 'rand' in match:
                        continue
                    iab_category,webname = match.split(',')
                    data.append({'Website Name': webname, 'IAB Category': iab_category})                                
                self.df4 = pd.DataFrame(data) 
                # Create a DataFrame from the table content
                # self.df4 = pd.read_csv(StringIO(table_content), skipinitialspace=True)

                # self.df4 = pd.DataFrame([row.split(':') for row in table_content], columns=['Category', 'Websites'])
                # self.df4 = self.df4.apply(lambda x: x.str.strip())

                # Extract the last paragraph
                self.pfl4 = paragraphs[-1] if match not in paragraphs[-1] else ""
            except Exception as error:
                print(error)
                self.pff4 = text
            yield
            print("News")
            reply5 = {"role":"assistant","content":f"{ind_web_response_msg['content']}"}
            message6 = {"role":"user","content":f"Suggest the right list of top 10 News website inventory to run the following brand ads along with the IAB categories. Return a table as comma-separated values with two columns - IAB Category and Website Name"}
            # # st.info(f'X-rae Response -  {ind_web_response_msg["content"]}')              
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            sysmessage,
                            message1,
                            reply1,
                            message2,
                            reply2,
                            message3,
                            reply3,
                            message4,
                            reply4,
                            message5,
                            reply5,
                            message6,
                    ],
                    request_timeout=20
                )          
                # import pdb;pdb.set_trace()
                news_response_msg = response["choices"][0]["message"]
                # self.news_response_msg = news_response_msg["content"]
                text = news_response_msg["content"]
                if "apologize" in text or 'Unfortunately' in text or "Apologize" in text or "Apologise" in text or "apologise" in text:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                                sysmessage,
                                message1,
                                reply1,
                                message2,
                                reply2,
                                message3,
                                reply3,
                                message4,
                                reply4,
                                message5,
                                reply5,
                                message6,
                        ],
                        request_timeout=20
                    )          
                    # import pdb;pdb.set_trace()
                    news_response_msg = response["choices"][0]["message"]
                    # self.news_response_msg = news_response_msg["content"]
                    text = news_response_msg["content"]
                print(text)
                paragraphs = text.split('\n\n')
                # Extract the first paragraph
                self.pff5 = paragraphs[0]

                # Extract the table content
                table_content = paragraphs[1]
                matches = table_content.split('\n')
                data = []
                for match in matches:
                    if 'IAB' in match or '---' in match or not match or 'rand' in match:
                        continue
                    iab_category,webname = match.split(',')
                    data.append({'Website Name': webname, 'IAB Category': iab_category})                                
                self.df5 = pd.DataFrame(data) 
                # Create a DataFrame from the table content
                # self.df5 = pd.read_csv(StringIO(table_content), skipinitialspace=True)

                # Extract the last paragraph
                self.pfl5 = paragraphs[-1] if match not in paragraphs[-1] else ""
            except Exception as error:
                print(error)
                self.pff5 = text
            yield
            print("News India")        
            reply6 = {"role":"assistant","content":f"{news_response_msg['content']}"}
            message7 =  {"role":"user","content":f"Suggest the right list of top 10 Indian News website inventory to run the following brand ads along with the IAB categories. Return a table as comma-separated values with two columns - IAB Category and Website Name"}             
            # # st.info(f'X-rae Response -  {news_response_msg["content"]}')     
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            sysmessage,
                            message1,
                            reply1,
                            message2,
                            reply2,
                            message3,
                            reply3,
                            message4,
                            reply4,
                            message5,
                            reply5,
                            message6,
                            reply6,
                            message7,
                    ],
                    request_timeout=20
                )          
                # import pdb;pdb.set_trace()
                news_ind_response_msg = response["choices"][0]["message"]
                # self.news_ind_response_msg = news_ind_response_msg["content"]
                text = news_ind_response_msg["content"]
                if "apologize" in text or 'Unfortunately' in text or "Apologize" in text or "Apologise" in text or "apologise" in text:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                                sysmessage,
                                message1,
                                reply1,
                                message2,
                                reply2,
                                message3,
                                reply3,
                                message4,
                                reply4,
                                message5,
                                reply5,
                                message6,
                                reply6,
                                message7,
                        ],
                        request_timeout=20
                    )          
                    # import pdb;pdb.set_trace()
                    news_ind_response_msg = response["choices"][0]["message"]
                    # self.news_ind_response_msg = news_ind_response_msg["content"]
                    text = news_ind_response_msg["content"]
                    print(text)
                paragraphs = text.split('\n\n')
                # Extract the first paragraph
                self.pff6 = paragraphs[0]

                # Extract the table content
                table_content = paragraphs[1]
                matches = table_content.split('\n')
                data = []
                for match in matches:
                    if 'IAB' in match or '---' in match or not match or 'rand' in match:
                        continue
                    iab_category,webname = match.split(',')
                    data.append({'Website Name': webname, 'IAB Category': iab_category})                                
                self.df6 = pd.DataFrame(data) 
                # Create a DataFrame from the table content
                # self.df6 = pd.read_csv(StringIO(table_content), skipinitialspace=True)

                # Extract the last paragraph
                self.pfl6 = paragraphs[-1] if match not in paragraphs[-1] else ""            
            except Exception as error:
                print(error)
                self.pff6 = text
            yield
            return True               
            # st.info(f'X-rae Response -  {news_ind_response_msg["content"]}')        
                    
        elif not os.path.exists(uploaded_file):
            self.alert_msg_header = "Error"
            self.alert_msg = "Failed to load image"
            self.alert_change()
            self.processing,self.complete = False,True
            yield
            # return rx.window_alert("Failed to load image")
            # st.error("Failed to load image")
        elif not article.title:
            self.alert_msg_header = "Error"
            self.alert_msg = "Failed to load data file"
            self.alert_change()
            self.processing,self.complete = False,True
            yield            
            # return rx.window_alert("Failed to load data file ")
            # st.error("Failed to load data file ") 
        else:
            self.alert_msg_header = "Error"
            self.alert_msg = "Unknown Error"
            self.alert_change()
            self.processing,self.complete = False,True
            yield            
            # return rx.window_alert("Failed to load data file ")
            # print(f"Files not found - {uploaded_file} - 'out.json'")      

    def alert_change(self):
        self.alert_show = not (self.alert_show)

def tag_list(tag: str):
    # index = random.randint(0,4)
    global colindex
    # import pdb;pdb.set_trace()
    if colindex == 0:
        colindex = 1
    else:
        colindex = 0
    print(f"Color index is {colindex}")

    colorval = ["#a61d55","#991BE2"]
    return rx.badge(
            tag, variant="solid",
            background="transparent",
            line_height="1.42",
            # bg="#fff",
            color=f"{colorval[colindex]}",##991BE2
            border_color=f"{colorval[colindex]}",##991BE2
            border_width="1px",
            border_style= "solid",
            font_size="1em",
            font_weight="normal",
            text_transform = "lowercase",
            border_radius = "1.41em",
            cursor = "pointer",
            # box_shadow = "5px 5px #000000",
            margin = "6px",
            padding = "0.7em 1.4em"
        )

def colored_box(color: str):
    return rx.box(rx.text(color), bg=color)



def index() -> rx.Component:
    return rx.fragment(
        rx.hstack(
            rx.image(src="https://citrusberry.biz/assets/img/menu_logo1.png", width="41px", height="auto"),
            rx.image(src="https://citrusberry.biz/assets/img/menu_logo.png", width="90px", height="auto"),
            padding="10px",
            margin="5px",
        ),
        # rx.color_mode_button(rx.color_mode_icon(), float="right"),
        rx.vstack(
            rx.tooltip(
                rx.card(  
                    rx.center(
                        rx.image(src="logo-no-background.png", width="200px", height="auto"),
                    #     rx.heading("X-Rae Output", size="xl", color="#fb5e78"),
                    # border_radius="15px",
                    # border_width="thick",
                    width="100%",
                    # border_color="#fb5e78",
                    ), 
                ),                       
                # background="linear-gradient(90deg, #ff5c72, #a485f2)",
            # rx.heading("Contextual AI Demo!", font_size="2em",color="#a61d55",),
            label="Please enter or select a link and click on Analyze",
            ),         
            rx.alert_dialog(
                rx.alert_dialog_overlay(
                    rx.alert_dialog_content(
                        rx.alert_dialog_header(State.alert_msg_header),
                        rx.alert_dialog_body(
                            State.alert_msg
                        ),
                        rx.alert_dialog_footer(
                            rx.button(
                                "Close",
                                on_click=State.alert_change,
                            )
                        ),
                    )
                ),
                is_open=State.alert_show,
            ),            
            # rx.box("Get started by editing ", rx.code(filename, font_size="1em")),
            rx.accordion(
                rx.center(        
                    rx.accordion_item(
                        rx.accordion_button(
                            rx.tooltip(
                                rx.text(
                                    "Select an example from dropdown"
                                ),
                            label="Click here to see Examples",
                            ),                                  
                            rx.accordion_icon(),
                        ),
                        rx.accordion_panel(
                            rx.center(
                                rx.tooltip(
                                    rx.select(
                                        a_options,
                                        placeholder="select.",
                                        # on_change=State.set_selected_option_a,
                                        # on_change=State.fetch_preloaded,
                                        on_change=State.set_text,
                                    ),
                                label="Please select a link and Click on Analyze",
                                ),                      
                                width="1000px"    
                            ),    
                        ),
                    ),
                ),
                allow_multiple=True,
                width="80%",
                font_size="16px",
                #  width="1000px",
            ),    
            rx.center(
                rx.tooltip(
                    rx.icon(
                            # tag="link",margin_right="10px",on_click=rx.set_clipboard(State.selected_option_a),
                            tag="link",margin_right="10px",
                            ),
                # label="Copy Link",
                label="Please enter a url link and click on Analyze",
                ),          
                rx.tooltip(            
                    rx.input(
                        # placeholder="Enter the page url",
                        on_blur=State.set_text,
                        width="100%",
                        value=State.text,
                        on_change=State.set_text,
                        # default_value=State.text,
                            
                    ),
                label="You can enter a custom url link and click on Analyze",
                ),    
                width="1000px"    
            ),                                   
            rx.hstack(
                # rx.button(
                #     "Clear", on_click=State.clear_text,width="100%",
                # ),  
                rx.html("""
                        <button class='btn-101'>
                        Analyse
                        <svg>
                            <defs>
                            <filter id='glow'>
                                <fegaussianblur result='coloredBlur' stddeviation='5'></fegaussianblur>
                                <femerge>
                                <femergenode in='coloredBlur'></femergenode>
                                <femergenode in='coloredBlur'></femergenode>
                                <femergenode in='coloredBlur'></femergenode>
                                <femergenode in='SourceGraphic'></femergenode>
                                </femerge>
                            </filter>
                            </defs>
                            <rect />
                        </svg>
                        </button>                      
                        """,on_click=State.run_analysis),
                # rx.button(
                #     "Analyze", on_click=State.run_analysis,is_loading=State.processing,width="100%",
                #     background_image="linear-gradient(90deg, #ff5c72, #a485f2)",
                # ),                
            ),  
            rx.cond(
                State.processing,
                rx.circular_progress(is_indeterminate=True,color="#a61d55"
                ),    
            ),                                  
            rx.cond(
                State.complete,
                    rx.responsive_grid(        
                     rx.vstack(
                        # rx.divider(border_color="#a61d55"),
                        rx.heading(State.article_title, size="lg",margin="30px",
                                   ),
                        # rx.hstack(
                        rx.responsive_grid(        
                            rx.card(
                                rx.center(
                                    rx.image(
                                    src=State.image_url,
                                    height="25em",
                                    width="37.5em",
                                    ), 
                                border_radius="10px",
                                border_width="2px",
                                border_color="#a61d55",
                                width="100%",
                                ),
                                header=rx.heading("Article Image", size="lg"),
                                # footer=rx.heading("Footer", size="sm"),
                            ), 
                            rx.card(
                                rx.text(State.article_text),
      
                                header=rx.heading("Article Text", size="lg"),
                                # footer=rx.heading("Footer", size="sm"),
                            ), 
                            columns=[2],
                            spacing="4",                                                                 
                        ),
                        # rx.divider(border_color="black"),  
                        rx.responsive_grid(
                            rx.card(  
                                rx.center(
                                    rx.vstack(
                                        rx.hstack(
                                            rx.foreach(State.model_tag_list,tag_list),rx.spacer(),
                                        ),
                                        rx.hstack(
                                            rx.heading(State.model_caption, size="lg", ),
                                        ),
                                    ),
                                ),
                                background="linear-gradient(45deg,#e6e4fc,#fceded)",
                                header=rx.heading("X RAE Image Analysis", size="lg"),
                                
                            ),
                            columns=[1],
                            spacing="4",    
                            width="100%",
                        ),  
                        rx.responsive_grid(
                            # rx.divider(border_color="black"),
                            rx.card(  
                                rx.center(
                                    rx.vstack(
                                        # rx.hstack(
                                        rx.heading(
                                            State.sff1+State.sentiment_disp_value, color=State.sentiment_filled_bg,opacity="0.8"
                                        ),
                                        # ),
                                        # rx.hstack(
                                        rx.progress(value=State.sentiment_value, width="100%",color_scheme=State.sentiment_color,height="15px",bg="#fff",opacity="0.8"),        
                                        # ),
                                        width="75%",
                                    ),
                                ),
                                background="linear-gradient(45deg,#e6e4fc,#fceded)",
                                header=rx.heading("Overall Sentiment", size="lg"),
                                
                            ),
                            columns=[1],
                            spacing="4",    
                            width="100%",
                        ),   
                        rx.tabs(
                            items=[
                                (
                                    "Targeting Metrics",
                                    rx.responsive_grid(  
                                        rx.card(
                                                rx.vstack(        
                                                    rx.data_table(
                                                        data=State.keyword_list,
                                                        pagination=False,
                                                        search=False,
                                                        sort=False,
                                                    ),      
                                                    rx.text(State.iab_safety_response_msg),
                                            

                                                ),
                                                header=rx.heading("Keywords", size="lg"),
                                                # footer=rx.heading("Footer", size="sm"),
                                            ), 

                                            rx.card(
                                                rx.cond(
                                                    State.error_occured,
                                                    rx.text(State.iab_response_msg),
                                                    rx.data_table(
                                                        data=State.df,
                                                        # pagination=True,
                                                        # search=True,
                                                        # sort=True,
                                                    ),      
                                                ),                                    
                                                header=rx.heading("IAB Categories", size="lg"),
                                                # footer=rx.heading("Footer", size="sm"),
                                            ), 
                                    columns=[2],
                                    spacing="4",               
                                    ),
                                ),
                                (
                                    "Brands", 
                                    rx.responsive_grid(  
                                    rx.card(
                                            rx.vstack(
                                                rx.text(State.pff1),
                                                rx.data_table(
                                                    data=State.df1,
                                                    # pagination=True,
                                                    # search=True,
                                                    # sort=True,
                                                ),             
                                                rx.text(State.pfl1,font_style="italic"),
                                            ),
                                            header=rx.heading("Global Brands To Target", size="lg"),
                                            # footer=rx.heading("Footer", size="sm"),
                                        ), 
                                        rx.card(
                                            # rx.text(State.indian_brands_response_msg),
                                            rx.vstack(
                                                rx.text(State.pff2),
                                                rx.data_table(
                                                    data=State.df2,
                                                    # pagination=True,
                                                    # search=True,
                                                    # sort=True,
                                                ),             
                                                rx.text(State.pfl2,font_style="italic"),
                                            ),                                
                
                                            header=rx.heading("Indian Brands To Target", size="lg"),
                                            # footer=rx.heading("Footer", size="sm"),
                                        ), 
                                        columns=[2],
                                        spacing="4",               
                                    ),
                            ),
                            (
                                "Website Inventory",
                                rx.responsive_grid(  
                                    rx.card(
                                        # rx.text(State.web_inventory_response_msg),
                                        rx.vstack(
                                            rx.text(State.pff3),
                                            rx.data_table(
                                                data=State.df3,
                                                # pagination=True,
                                                # search=True,
                                                # sort=True,
                                            ),             
                                            rx.text(State.pfl3,font_style="italic"),
                                        ),                                
            
                                        header=rx.heading("Website Inventory to target", size="lg"),
                                        # footer=rx.heading("Footer", size="sm"),
                                    ), 
                                    rx.card(
                                        # rx.text(State.ind_web_response_msg),
                                        rx.vstack(
                                            rx.text(State.pff4),
                                            rx.data_table(
                                                data=State.df4,
                                                # pagination=True,
                                                # search=True,
                                                # sort=True,
                                            ),             
                                            rx.text(State.pfl4,font_style="italic"),
                                        ),                                
            
                                        header=rx.heading("Indian Website Inventory to target", size="lg"),
                                        # footer=rx.heading("Footer", size="sm"),
                                    ), 
                                    columns=[2],
                                    spacing="4",               
                                ),  
                            ),
                            (
                                "News Inventory",
                                rx.responsive_grid(  
                                    rx.card(
                                        # rx.text(State.news_response_msg),
                                        rx.vstack(
                                            rx.text(State.pff5),
                                            rx.data_table(
                                                data=State.df5,
                                                # pagination=True,
                                                # search=True,
                                                # sort=True,
                                            ),             
                                            rx.text(State.pfl5,font_style="italic"),
                                        ),                                
            
                                        header=rx.heading("News Website Inventory to target", size="lg"),
                                        # footer=rx.heading("Footer", size="sm"),
                                    ), 
                                    rx.card(
                                        # rx.text(State.news_ind_response_msg),
                                        rx.vstack(
                                            rx.text(State.pff6),
                                            rx.data_table(
                                                data=State.df6,
                                                # pagination=True,
                                                # search=True,
                                                # sort=True,
                                            ),             
                                            rx.text(State.pfl6,font_style="italic"),
                                        ),      
                                        header=rx.heading("Indian News Website Inventory to target", size="lg"),
                                        # footer=rx.heading("Footer", size="sm"),
                                    ), 
                                    columns=[2],
                                    spacing="4",               
                                ),                                                
                            )
                            ],
                        ),                                                                       
                     ),
                    animation="fadeInAnimation ease 3s",

                    )
            ),            
            spacing="1.5em",
            font_size="1em",
            padding="3%",
            shadow="lg",
            border_radius="lg",            
        ),
        width="100%",
        height="auto",
        #        
    )

def about():
    return rx.text("About Page")



# Add state and page to the app.
app = rx.App(state=State,stylesheets=[
        "styles/fontstyles.css","styles/center-simple.css","styles/introjs.min.css"  # This path is relative to assets/
    ],style=style,scripts="intro.js")
app.add_page(index,title="Contextual Demo")
app.add_page(about, route="/about")
app.compile()


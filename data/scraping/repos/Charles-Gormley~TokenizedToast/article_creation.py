import openai
import base64
from time import sleep
from obtain_keys import load_api_keys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 

api_keys = load_api_keys()

openai.api_key = api_keys['OPEN_AI']

MAX_RETRIES = 5

###### Article Creation ######
def create_title(article:str, preferences:str) -> str:
    attempt = 0
    while attempt < MAX_RETRIES:
        try: 
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": f"Create a title for this article with the following preferences: {preferences}. Keep only title in response, should be 1 to 8 words"},
                    {"role": "user", "content": article }
                ],
                temperature=0.82,
                max_tokens=20,
                top_p=.9,
                frequency_penalty=0,
                presence_penalty=0
            )
            output = response['choices'][0]['message']['content']
            stripped_output = strip_quotes(output)
            return stripped_output
        except:
            openai.api_key = api_keys['OPEN_AI']
            attempt += 1
            sleep(5)



def summarize(article:str, preferences:str) -> str:
    attempt = 0
    while attempt < MAX_RETRIES:
        try: 
            response = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo-16k",
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": f"You are a newsletter writer tailoring articles for your user with these preferences: {preferences}. Try to keep the content summmary around 250 words and inlcude nothing else in the response. Keep only the news in the summary rather than educating individuals on the basics."},
                    {"role": "user", "content": article }
                ],
                temperature=0.82,
                max_tokens=500,
                top_p=.9,
                frequency_penalty=0,
                presence_penalty=0
            )
        except:
            openai.api_key = api_keys['OPEN_AI']
            attempt += 1
            sleep(5)

    output = response['choices'][0]['message']['content']
    stripped_output = strip_quotes(output)
    return stripped_output
        

def create_email_subject(titles:list, preferences:str) -> str:
    attempt = 0
    while attempt < MAX_RETRIES:
        try: 
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                        {"role": "system", "content": f"Create an email subject line to catch the users attention and describe the list of articles with the following preferences: {preferences}. Keep only the email subject in your respoonse, which should be 1 to 8 words. This should aim to have a high open rate."},
                        {"role": "user", "content": str(titles)}
                        ],
                temperature=0.82,
                max_tokens=20,
                top_p=.9,
                frequency_penalty=0,
                presence_penalty=0
            )
            output = response['choices'][0]['message']['content']
            stripped_output = strip_quotes(output)

            return stripped_output
        except:
            openai.api_key = api_keys['OPEN_AI']
            attempt += 1
            sleep(5)

def create_section_header(titles:list) -> str:
    attempt = 0
    while attempt < MAX_RETRIES:
        try: 
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                        {"role": "system", "content": f"Create a section header for the following titles. Keep only the section header in the response which should be 1 word. "},
                        {"role": "user", "content": str(titles)}
                        ],
                temperature=0.82,
                max_tokens=20,
                top_p=.9,
                frequency_penalty=0,
                presence_penalty=0
            )
            output = response['choices'][0]['message']['content']
            stripped_output = strip_quotes(output)

            return stripped_output
        except:
            openai.api_key = api_keys['OPEN_AI']
            attempt += 1
            sleep(5)
    
def create_header_image(titles:list) -> str:
    attempt = 0
    while attempt < MAX_RETRIES:
        try: 
            client = openai.OpenAI(api_key=api_keys['OPEN_AI'])
            response = client.images.generate(
                model="dall-e-3",
                prompt=f"Create an Image Header for a newsletter with these titles: {str(titles)}",
                size="1024x1024",
                quality="standard",
                n=1,
            )
            return response.data[0].url
        except:
            openai.api_key = api_keys['OPEN_AI']
            attempt += 1
            sleep(5)

    

def create_content(article:str, preferences:str) -> dict:
    content_dict = dict()
    content_dict['title'] = create_title(article, preferences)
    content_dict['body'] = summarize(article, preferences)
    return content_dict

###### Formatting ######
light_mode = '''
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: transparent;
            color: white;
            padding: 20px;
        }

        .container {
            max-width: 80%;
            margin: 0 auto;
            border: 2px solid white;
            padding: 20px;
            border-radius: 15px;
        }

        .section {
            background-color: white;
            color: black;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 10px;
            border: 2px solid #555;
        }

        .section img {
            display: block;
            max-width: 100%;
            height: auto;
            margin: 0 auto 10px auto;
        }

        .header-box {
            background-color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 10px;
            border: 2px solid #555;
        }

        .header {
            text-align: center;
        }

        .header img {
            display: block;
            width: 500px;
            margin: 0 auto 10px auto;
        }

        .header h1 {
            font-size: 24px;
            color: black;
            padding: 10px 20px;
            border-radius: 10px;
        }

        .article {
            background-color: white;
            color: black;
            padding: 20px 30px;
            margin-bottom: 30px;
            border-radius: 10px;
            border: 2px solid #555;
        }

        .author {
            display: flex;
            align-items: center;
        }

        .author:before {
            content: '✍️';
            margin-right: 10px;
        }

        h2 {
            font-size: 22px;
            border-bottom: 2px solid #555;
            padding-bottom: 10px;
            color: #0077b6;
        }

        .market {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: white;
            color: black;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 10px;
            border: 2px solid #555;
        }

        .percentage.green {
            color: green;
            background-color: #dafac5;
            padding: 5px;
            border-radius: 5px;
        }

        .percentage.red {
            color: red;
            background-color: #f9d6d5;
            padding: 5px;
            border-radius: 5px;
        }

    </style>'''

def strip_quotes(text):
    return text.replace('"', '').replace("'", "")


def format_section(passages:list, header:str, header_img_url:str) -> str:
    section_html = f'''
    <div class="section">
        <h2>{strip_quotes(header)}</h2>
        <img src="{header_img_url}" alt="Section Image">
        {passages[0]}
        {passages[1]}
    </div>
    '''
    return section_html

def format_passage(content_dict:dict) -> str:
    html = f'''
    <h3>{content_dict['title']}</h3>
    <div class="article">
        <p>{content_dict['body']}</p>
    </div>
    '''
    return html

def create_email_html(content:list, preferences:str, style:str) -> dict:
    email = dict()
    
    passages = []
    sections = []
    titles = []
    counter = 0

    for content_dict in content:
        passage = format_passage(content_dict)
        passages.append(passage)
        titles.append(content_dict['title'])
        counter += 1
        if counter % 2 == 0:
            header = create_section_header(titles=titles[-2:])
            image_url = create_header_image(titles=titles[-2:])
            section = format_section(passages=passages[-2:], header=header, header_img_url=image_url)
            sections.append(section)
    
    
    
    # Combine passages
    combined_sections = ''.join(sections)
    
    # Generate final email HTML
    email_html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        {light_mode}
    </head>
    <body>
        <div class="container">
            <div class="header-box">
                <div class="header">
                    <img src="https://toast-logos-newsletter.s3.amazonaws.com/ToastLogo-removebg-preview.png" alt="Logo">
                    <h1>Here's your Morning Toast</h1>
                </div>
            </div>
            {combined_sections}
        </div>
    </body>
    </html>
    '''
    
    email['subject'] = create_email_subject(titles, preferences)
    email['body'] = email_html

    return email

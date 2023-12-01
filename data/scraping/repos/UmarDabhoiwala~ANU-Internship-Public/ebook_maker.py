import openai
from halo import Halo
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from ebooklib import epub
from faker import Faker 
import markdown
import random 
import datetime
from bs4 import BeautifulSoup
fake = Faker()

openai.api_key = config.OPENAI_API_KEY
spinner = Halo(text='Loading', spinner='dots')

def chat_gpt_completion(chat_message, append = False, usr_prompt = ""):
    
    spinner.start("text generating")
        
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages= chat_message
    )
    
    spinner.succeed("text generated")
    
    response = completion['choices'][0]["message"]["content"]
    
    if append:
        chat_message.append({"role": "system", "content": response})
        
    
    return response, chat_message

system_prompt = """You are the famous short story author {author_name}, renowed for thier ability to explore complex human themes in just a couple pages. 
Today you will write a few more of your tales creating stories based on what the user inputs. Respond in markdown format make sure to include a title
"""

def write_story(Genre, Style, Setting, Basic = True, debug = False, message = []):
    
    if Basic: 
        story_prompt = f"""Write a {Genre} short story in the style of {Style} set in {Setting}"""
    else: 
        story_prompt = f"""Write a {Genre} short story in the style of {Style} set in {Setting}. Make the stories at least 500 words long"""
    
    if message == []:
        message = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": story_prompt}]
    else: 
        message.append({"role": "user", "content": story_prompt})
        
    response, m = chat_gpt_completion(message, append=True)
    html_content = markdown.markdown(response)

    if debug == True:
        now = datetime.datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        output_html_file = f"output_file_{date_time_str}.html"   
        with open(output_html_file, "w", encoding="utf-8") as html_file:
            html_file.write(html_content)
        
    return html_content, message

def extract_heading(content):

    soup = BeautifulSoup(content, "lxml")

    heading = soup.find("h1")
    
    if heading:
        return heading.text.strip()
    else:
        return None


def create_book(num_stories, author_name, Genre, Style, Setting):
        
    book = epub.EpubBook()

    book.set_identifier("id123456")
    book.set_title(f"{author_name}'s short stories")
    book.set_language('en')
    book.add_author(author_name)
    
     # add default NCX and Nav file
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    
    # Add cover page
    cover_html = f'''
    <html>
    <head>
        <title>Cover</title>
        <link rel="stylesheet" href="ebook-style.css" type="text/css"/>
    </head>
    <body>
        <h1>{author_name}'s short stories LLLLL</h1>
    </body>
    </html>
    '''

    cover = epub.EpubHtml(title='Cover', file_name='cover.xhtml')
    cover.set_content(cover_html)
    book.add_item(cover)
    

    
    d = {}
    filenames = []
    toc = []
    spine = ["cover","nav"]
    message = []
    for x in range(0,num_stories):
        
        html_content, message = write_story(Genre, Style, Setting, Basic = False, debug = False, message = message)
        heading = extract_heading(html_content)
        if heading == None:
            random_num = str (random.randint(0, 100000))
            heading = "Untitled" + random_num
            
        filename = f'{heading}.xhtml'
        filenames.append(filename)
        d[f"chapter{x}"] = epub.EpubHtml(title=heading,
                                                file_name=filename,
                                                lang='en')
        d[f"chapter{x}"].set_content(html_content)
        
      
        toc.append(d[f"chapter{x}"])
        spine.append(d[f"chapter{x}"])
        book.add_item(d[f"chapter{x}"])
        
    # define CSS style
    style = '''
    body {
        font-family: "Helvetica", "Arial", sans-serif;
        font-size: 1em;
        line-height: 1.6;
        color: #333;
        background-color: #fff;
        margin: 25px 50px 75px 100px;
        padding: 0;
    }
    h1 {
        font-size: 2em;
        text-align: center;
        padding: 1em 0;
    }
    '''
    nav_css = epub.EpubItem(
        uid="style_nav",
        file_name="ebook-style.css",
        media_type="text/css",
        content=style,
    )

    # add CSS file
    book.add_item(nav_css)
        
    book.toc = ((epub.Section('Stories'), toc), )
    book.spine = spine

    epub.write_epub("files/book.epub", book, {})
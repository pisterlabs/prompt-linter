import requests
import json
from bs4 import BeautifulSoup
import openai

class Users:
    start = 0
    end = 5
    val = ''

    def __init__(self, email = None, sesid = None):
        self.email = email
        self.sesid = sesid
        self.api_key = ''
        self.book_name = ''
        self.book_ext = ''
        self.response = None
        self.soup = None
        self.div_tag = None
        self.total_books = 0
        self.books = []
        self.book_dict = {}
        self.currbook = []
        self.remember_index = []
        self.gotbook = False
        self.uid = ''
        
        
    def set_userid (self, uid):
        self.uid = uid
        
    def get_userid (self):
        return self.uid    
        
    def get_book(self):
        return self.book_name
    
    def set_book(self, book_name):
        self.book_name = book_name
    
    def set_val (self, val):
        self.val = val
        
    def get_val (self):
        return self.val
    
    def get_email (self):
        return self.email
    
    def gpt_intent(self, book_index_list, choice):
        if choice==1:
            return "Tell me more about book " + str(self.book_dict[book_index_list[0]]['Name'])
        elif choice==2:
            return "Compare the following books " + str(self.book_dict[book_index_list[0]]['Name']) + " and " + str(self.book_dict[book_index_list[1]]['Name']) + " and contrast the differences between both briefly"
        
    def gpt_response(self, book_index_list, choice):
        openai.api_key = ""
        response = openai.Completion.create(
          engine="text-davinci-003",
          prompt=self.gpt_intent(book_index_list,choice),
          temperature=0.7,
          max_tokens=250,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
    
        return str (response['choices'][0]['text'])
    
    def get_link(self, main_link):
        response = requests.get(
            url='https://proxy.scrapeops.io/v1/',
            params={
                'api_key': '',
                'url': main_link, 
            },
        )
    
        soup = BeautifulSoup(response.content, "html.parser")
        name_tags = soup.find_all("ul")
        options = []
        
        for name_tag in name_tags:
            a_tags = name_tag.find_all('a', class_='js-download-link')
            for a_tag in a_tags:
                if 'href' in a_tag.attrs:
                    href = a_tag['href']
                    options.append(href)
        options=options[2:len(options)-1]
        bl_link = '\n'.join(f"{i+1}. {link}" for i, link in enumerate(options))
        return str(bl_link)
    
    def scraped_book_dict(self, book_name, ext):
        self.book_name = book_name
        self.ext = ext
        self.response = requests.get(
            url='https://proxy.scrapeops.io/v1/',
            params={
                'api_key': self.api_key,
                'url': f"https://annas-archive.org/search?&ext={self.ext}&q={self.book_name}", 
            },
        )
        self.soup = BeautifulSoup(self.response.content, "html.parser")
        self.div_tag = self.soup.find('div', {'class': 'h-[125]'})
        self.total_books=0
        self.books=[]   
        for item in list(self.div_tag.parent):
            if not(item=='\n'):
                self.total_books+=1
                self.books.append(item)
        self.books.pop(0)
        for i in range(self.total_books - 1):
            self.book_dict[i]={}
            cur_book=self.books[i]
            try:
                html = str(cur_book)
                soup = BeautifulSoup(html, 'html.parser')
                href = soup.find('a')['href']
                self.book_dict[i]['href'] = "https://annas-archive.org" + href
            except Exception:
                try:
                    string = str(self.books[i]).split('\n')[1]
                    href = re.search(r'href="([^"]+)"', string)
                    if href:
                        self.book_dict[i]['href'] = "https://annas-archive.org" + href.group(1)
                    else:
                        self.book_dict[i] = {'Name': "Book Unavailable", 'href': "Link Unavailable"}
                except Exception:
                    self.book_dict[i] = {'Name': "Book Unavailable", 'href': "Link Unavailable"}
            try:
                h3_tag = soup.find('h3')
                self.book_dict[i]['Name']=h3_tag.text
            except Exception:
                if h3_tag is None:
                    html = str(cur_book)
                    soup = BeautifulSoup(html, 'html.parser')
                    html = soup.prettify()
                    start = html.find('<!--')
                    end = html.find('-->')
                    commented_out_html = html[start:end]
                    soup = BeautifulSoup(commented_out_html, 'html.parser')
                    h3_tag = soup.find('h3')
                    try:
                        self.book_dict[i]['Name'] = h3_tag.text
                    except Exception:
                        self.book_dict[i] = {'Name': "Book Unavailable", 'href': "Link Unavailable"}
        return self.book_dict
        
    def return_structured_response(self, val):
    
        response = {
            'dialogAction': {
                'type': 'Close',
                'fulfillmentState': 'Fulfilled',
                'message': {
                    'contentType': 'PlainText',
                    'content': val
                }
            }
        }
    
        return response
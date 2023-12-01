# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import json
import time
import config
openai.api_key = config.OPENAI_KEY
from bookgen_topdf import gen_pdf




def save(filename, text):
    f = open(filename, 'w+')
    f.write(text)
    f.close()
    
def generate_chapters(book_title):
    prompt = [{"role": "system", "content":  f"generate a list of chapters and subchapters for a book titled {book_title} in json format. do not include any explanation or code formatting. format it in this way: "+"{\"chapter_name\":[\"subchapter_names\"],}"+". please include between 5 and 10 subchapters per chapter. use this format exactly."},{"role": "user", "content":  "generate with 4 space indents"},]
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-0301",
      messages=prompt
    )
    
    gpt_response = response['choices'][0]['message']['content']
    
    return gpt_response

def generate_content(chaptername, subchapter, book_title):
    prompt = [{"role": "system", "content":  f"generate the content for a subchapter in a book. the chapter title is {chaptername}. the title of the subchapter is {subchapter}. the title of the book is {book_title}. please only include the requested data. "},{"role": "user", "content":  "do not include the chapter title, the subchapter title, or the book title in the data, only the chapter content." },]
    
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-0301",
      messages=prompt
    )
    time.sleep(3)
    gpt_response = response['choices'][0]['message']['content']
    
    return gpt_response

if __name__ == '__main__':
    book_title = input("please input a book title: ")
    
    chapters_names = generate_chapters(book_title)
    save('chapters.json', chapters_names)
    
    chapters_json = json.loads(chapters_names)

    for c in chapters_json:
        print(c)
            
    chapters = []
    
    for chapter_name, subchapters in chapters_json.items():
        print(chapter_name)
        sections = []
        for subchapter_name in subchapters:
            
            while True:
                try:
                    content = generate_content(chapter_name, subchapter_name, book_title)
                    break
                except openai.error.RateLimitError:
                    print('Ratelimit Trying again in 5 minutes')
                    time.sleep(5*60)
                    
            section = {
                "subchapter": subchapter_name,
                "content": content
        }
            sections.append(section)
        chapter = {
            "chapter_title": chapter_name,
            "subchapters": sections
        }
        chapters.append(chapter)
    
    
    book_out = book_title.replace(' ', '_').replace(':','-').replace('?','')+ '.json'
    save(book_out, json.dumps(chapters, indent=4).replace('\u2019', '\'').replace('\u201c','\"').replace('\u201d','\"'))
    
    gen_pdf(book_title,chapters)
    

    
    
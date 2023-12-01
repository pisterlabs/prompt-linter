
import tiktoken
import math
import docx
import openai
import os
from dotenv import load_dotenv


load_dotenv('python.env')
openai.api_key = os.getenv('OPENAI_API_KEY')



#global variable defining the code object for switching into tokens
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")



#helper function for split_text() to decode the section_tokens back into words whether or not
#section_tokens is a list or a list of lists
def decode_text(token_list):
    if all(isinstance(i, list) for i in token_list):
        #the [] define it as a list, we run the function for each item in a list, and each time it does that, it creates a new item in the list
        texts = []
        for item in token_list:
            text = encoding.decode(item)
            texts.append(text)
    else: texts = encoding.decode(token_list)

    return texts

def split_text(text, max_token_length):
    token_list = encoding.encode(text)
    token_list_length = len(token_list)
    section_tokens = []
    section = []
    if token_list_length >= max_token_length:
        needed_sections = math.ceil(token_list_length/max_token_length)
        section_length = math.ceil(token_list_length/needed_sections)
        #create a bit of overlap in the sections 
        overlap = 50
        for i in range(0, token_list_length, section_length):
            tokens = (token_list[i:i+section_length + overlap])
            section_tokens.append(tokens)
            
    else: section_tokens = token_list


    section = decode_text(section_tokens)
    return section
        
    
def get_chapters(path, max_token_length):
    document = docx.Document(path)
    #initialize everything to avoid the none object errors
    sections = {}
    current_section = ""
    current_text = ""
    #keep in mind that the split_text function currently returns a list of tokens that needs to be
    #decoded into normal words before being fed back into the api -- this should be changed later
    #but my first attempt ran into a bug and I don't feel like fixing it right now when sticking the decode later
    #will work even though it feels like bad practice/ increasing technical debt
    for paragraph in document.paragraphs:
        if paragraph.style.name == 'Heading 1':
            if current_section != "":
                sections[current_section] = split_text(current_text, max_token_length)

                current_section = paragraph.text.strip()
                current_text = ""
            else:
                current_section = paragraph.text.strip()

        else:
            current_text += paragraph.text + " "
    
    sections[current_section] = split_text(current_text, max_token_length)

            
    return sections

##########
#We'll write the api call in this section

def create_new_file(filename):
    base_filename, extension = os.path.splitext(filename)
    new_filename = ""
    if not os.path.isfile(filename):
        open(filename, 'w').close()
        print(f"Created file: {filename}")
    else:
        i = 1
        while True:
            new_filename = f"{base_filename}-{i}{extension}"
            if not os.path.isfile(new_filename):
                open(new_filename, 'w').close()
                print(f"Created file: {new_filename}")
                break
            i += 1
    if new_filename:
        filename = new_filename
    return filename



def openai_create_summary(docx_path, title):
    #maybe it makes sense to just have the token length here?
    max_token_length = 3800
    system_message = "Given the following chapter from a novel, please provide a summary in less than a fifty words. Ensure the summary is accurate, concise, and doesn't include any invented details."
    novel = get_chapters(docx_path, max_token_length)
    summary = []
    output_token_max = 100
    
    #.items() calls up the tuple pair, and the key, value are naming the first and the second item in the
    #tuple pair for use elsewhere in the function. The name of the var does not matter, except that it would be
    #bad practice to give them a differnt one
    #so I can change from the suggested key, value title to chapter_name, text without it changing anything

    novel_chapters = list(novel.values())
    #flatten the list if one of the chapters were broken up by split_text
    novel_scenes = []
    for chapter in novel_chapters:
        if isinstance(chapter, list):
            for scene in chapter:
                novel_scenes.append(scene)
        else: 
            novel_scenes.append(chapter)

    filename = title + ' Summary' + '.txt'
    i = 1

    for scene in novel_scenes:
        
        try:
            scene_summary = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature = .3,
                max_tokens = output_token_max,

                messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": scene}
                ])
            
            summary.append(scene_summary)
            print (f'finished scene {i}')
            i += 1

        #need to figure out if the overload error is leading to repeating the loop
        #or much more likely skipping, and if skipping to set up a delay and retry
        except Exception as e:
            print(f" An error occurred: {e}")
            continue

    summary_text = ""
    for item in summary:
        summary_text += item['choices'][0]['message']['content'] + '\n'
    
    filename = create_new_file(filename)
    with open(filename, 'w', encoding='utf-8') as file:
         file.write(summary_text)



path = 'Mr Collins Widow.docx'
title = 'Reader I Married Him'
openai_create_summary(path, title)
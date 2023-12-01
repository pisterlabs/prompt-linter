import cohere , os
import datetime

# TODO put on virtual env
key = os.getenv('COHERE_KEY')

def co_command(model='command', prompt='', co : cohere.Client = None):
    if co == None:
        print("Error - Cohere client not defined.")
        return 
    
    response = co.generate(
        model=f'{model}',
        prompt=f'{prompt}',
        max_tokens=716,
        temperature=0.9,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE')

    # string = response.generations[0].text
    return response

def co_chat(model='command', prompt='', historic =[{}], co : cohere.Client = None):
    if co == None:
        print("Error - Cohere client not defined.")
        return 

    response = co.chat( 
        model=f'{model}',
        message=f'{prompt}',
        temperature=0.3,
        # chat_history=historic,


        # citation_quality='accurate',
        connectors=[{"id":"web-search"}],
        # documents=[] 
        ) 

    return response


def co_summarize(prompt='', co : cohere.Client = None):
    '''To use the summarized outpute, call the method summary from this object returned:
    
    >>> response = co_summarize(...).summary'''
    response = co.summarize( 
    text=f'{prompt}',
    length='auto',
    format='auto',
    model='command',
    temperature=0.3,
    ) 
    # print('Summary:', response.summary)
    return response


def get_song_array_old(response : str, debug = False):
    """ OUTPUT FORMAT: 
     any text..
     ...
     1- "Song name in quotes" by Author Name
    """
    if response == "":
        raise Exception("String passed as argument is null")

    div = response.split('\n')
    final = []

    # go through each split element to separate song name and author
    for s in div:
        if s == "" or len(s) ==0 :
            continue

        if s[0] == "-" or len(s[0]) ==0 or s[0] == '\n':
            try:
                initial_index = s.index("\"")
                final_index = s.index("\" ")
                song_name = s[initial_index+1:final_index]
                
                final_index = s.index("\" by ")

                author = s[final_index+5:]
                final.append([song_name, author])

            except ValueError:
                continue

    if debug:
        print(final)

    # if final string is not correctly formatted
    if final == []:
        return False

    return final



def get_song_array(response : str, debug = False):
    """ OUTPUT FORMAT: 
     any text..
     ...
     ..Name: "Song name in quotes or not"; Author: Author Name
    """
    if response == "":
        raise Exception("String passed as argument is null")

    div = response.split('\n')
    final = []

    name_separator = "Name: "
    author_separator = "Author: "

    # go through each split element to separate song name and author
    for s in div:
        if s == "" or len(s) ==0 or s[0] == '\n':
            continue

        else:
            try:
                initial_index = s.index(name_separator) + len(name_separator)
                final_index = s.index(";")
                song_name = s[initial_index:final_index]
                try:
                    song_name = song_name.split('\"')[1]
                
                except ValueError:
                    if(len(song_name)!=1 and debug):
                        print(f"error parsing: {song_name}")
            
                
                
                # final_index = s.index("\" by ")

                author_index = s.index(author_separator) + len(author_separator)
                author = s[author_index:]
                final.append([song_name, author])

            except ValueError:
                continue

    if debug:
        print(final)

    # if final string is not correctly formatted
    if final == []:
        return False

    return final


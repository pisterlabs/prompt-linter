import os
import openai
import json
import requests
from config import OPENAI_API_KEY, SEARCH_API_KEY, ENGINE_ID
from colorama import init, Fore, Style
from tools import write_response, write_messages, read_messages, num_tokens_from_messages, extract_text_from_url, parse_data
import time


init(autoreset=True)

openai.api_key = OPENAI_API_KEY

cse_api_key = SEARCH_API_KEY
search_engine_id = ENGINE_ID

outline = ""

def parse_history():
    messages = read_messages("plannerlog")
    #save content of last message as response
    response = messages[-1]["content"]
    if "[RESEARCHER]" in response:
        messages = read_messages("researcherlog")
        response = messages[-1]["content"]
        if "[SEARCH]" in response:
            #check search log
            a = 1
    if "[WRITER]" in response:
        read_messages("writerlog")
    if "[EDITOR]" in response:
        read_messages("editorlog")

    return

def google_custom_search(query, cse_api_key, search_engine_id):
    # Base URL for the API
    api_url = "https://www.googleapis.com/customsearch/v1"

    # Set the API parameters
    params = {
        "key": cse_api_key,
        "cx": search_engine_id,
        "q": query
    }

    # Send the request to the API
    response = requests.get(api_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON data
        json_data = response.json()

        # Extract the search results
        items = json_data.get("items", [])

        # Extract and return only the 'title' and 'link' from the search results
        results = [{"title": item["title"], "link": item["link"]} for item in items]
        #limit results to 7
        results = results[:7]

        for result in results:
            response = Reader(result)
            # Write response to a file
            write_response(response, "readerlog")

            # Extract the first JSON object from the response string
            #dictresponse = extract_first_json_obj(str(response))
            dictresponse = parse_data(response)

            # If a JSON object is found, load it and update the result dictionary
            if dictresponse and "Summary" in dictresponse and "Score" in dictresponse:
                result["summary"] = dictresponse["Summary"]
                result["score"] = dictresponse["Score"]
            else:
                break


        return results
    else:
        print(f"Request failed with status code {response.status_code}")
        return '{"title": "Your search failed.", "link": ""}'

def generate_response_gpt3(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature = 0.4
    )
    return response['choices'][0]['message']['content'], response['choices'][0]['finish_reason']

def generate_response(messages):
    max_retries=10
    initial_wait_time=1
    retries = 0
    response = None
    content = None
    finish_reason = None

    while retries <= max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0.7,
                messages=messages,
            )
            content = response['choices'][0]['message']['content']
            finish_reason = response['choices'][0]['finish_reason']
            break
        except openai.error.RateLimitError as e:
            if retries == max_retries:
                print("Exceeded maximum retries. Exiting.")
                return None, None
            else:
                wait_time = initial_wait_time * (2 ** retries)
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds.")
                time.sleep(wait_time)
                retries += 1

    return content, finish_reason

def Planner(user_input):
    global outline

    #store planner.txt into content
    with open("system.txt", "r") as f:
        plannercontent = f.read()
    messages = [
        {"role":"system", "content":f'{plannercontent}'},
        #get user input at some point in time, how we do this is undecided
        {"role":"user", "content":user_input} #TEST TEST TEST
        ]
    response =""
    #add something to know when to stop
    calls = 0
    while "[DONE]" not in response:
        calls += 1
        print(Fore.RED + f"Planner is working, call number {calls}")
        response,finish_reason = generate_response(messages)
        messages.append({"role":"assistant", "content":response})
        write_messages(messages, "plannerlog")
        #write messages to plannerlog in conversations folder
        print(response)

        #if && in response, find second && and store everything between the two && as the outline
        if "&&" in response:
            index = response.find("&&")
            next_index = response.find("&&", index + 1)
            outline = response[index + 2:next_index]



        
        #beginning of agent calls
        researcher_calls = 0
        if "[RESEARCHER]" in response:
            #will be format of [RESEARCHER]:"[topics]", get topics in between the two quotes, store as topics
            topics = response.split("[RESEARCHER]:")[1]
            if topics[0] == '"':
                topics = topics[1:-1]
            
            top_sources = Researcher(topics)
            researcher_calls+=1
            #file_name = "sources.txt"
            #save top_sources as a text file, increment the number at the end of the file name
            #with open(f"conversations/" + file_name, "w+") as f:
                #for source in top_sources:
                    #f.write(source["title"] + " | " + source["link"] + "\n" + "     " + source["summary"] + "\n\n")
            print(Fore.RED + "Researcher done researching")
            messages.append({"role":"user" , "content":f"[RESEARCHER]: Here is the list of sources: \n {top_sources}"})


        if "[WRITER]" in response:
            #will be format of [WRITER]: "[num]", get num in between the two quotes, store as num
            
            sources = response.split("[WRITER]:")[1]
            #find the first quote
            index = sources.find('"')
                #then the next character is a number
            num = int(sources[index + 1])
            section = Writer(num)
            messages.append({"role":"user", "content":f"[WRITER]: Here is the finished section: \n {section}"})

        if "[EDITOR]" in response:
            changes = Editor()
            print(changes)
            messages.append({"role":"user", "content":f"[EDITOR]: Here are the changes: \n {changes}"})



    return

def Researcher(topics):
    global outline
    #store researcher.txt into content
    with open("researcher.txt", "r") as f:
        researchercontent = f.read()
    messages = [
        {"role":"system", "content":f'{researchercontent}'},
        {"role":"user", "content":f'Search Topics:{topics}\n\n Outline: {outline}'}
        ]
    print(Fore.RED + "Researcher is working")
    response,a = generate_response(messages)
    messages.append({"role":"assistant", "content":response})
    write_messages(messages, "researcherlog")
    print(response)
    all_sources = []
    while "[RETURN]" not in response:
        if "[SEARCH]" in response:
            search_query = response.split("[SEARCH]:")[1]
            #remove the quotes from the search query
            #find the first quote
            index = search_query.find('"')
            #find the second quote
            next_index = search_query.find('"', index + 1)
            #store everything between the two quotes as the search query
            search_query = search_query[index + 1:next_index]

            search_results = google_custom_search(search_query, cse_api_key, search_engine_id)
            all_sources.extend(search_results)
            print(search_results)
            #write all_sources to a file

            print(Fore.RED + "Researcher is working")
            messages.append({"role":"user", "content":f"Search Results: {search_results}"})
            response,a = generate_response(messages)
            messages.append({"role": "assistant", "content": response})
            write_messages(messages, "researcherlog")
            print(response) #remove later on
    
    #return only top 10 all_sources sorted by score
    all_sources = sorted(all_sources, key=lambda i: i.get('score', 0), reverse=True)
    all_sources = all_sources[:10]
    file_name = "sources.txt"
    #save all_sources as a text file, increment the number at the end of the file name
    with open(f"conversations/" + file_name, "w+") as f:
        for source in all_sources:
            f.write(source.get("title", "") + " | " + source.get("link", "") + "\n" + "     " + source.get("summary", "") + "\n\n")
    return all_sources

def Reader(result):
    global outline
    #store reader.txt into content
    with open("reader.txt", "r") as f:
        readercontent = f.read()

    page_content = extract_text_from_url(result["link"])

    if page_content == "":
        print(Fore.RED + "Something failed")
        return '{"title": "Your search failed.", "link": ""}'


    messages = [
        {"role":"system", "content":f'{readercontent}'},
        {"role":"user", "content":f'Outline: {outline}\n\n Source: {page_content}'}
        ]

    tokens = num_tokens_from_messages(messages)
    over_prop = tokens / 4096 #8192
    if over_prop > 1:
        print(Fore.BLUE + "Truncating source")
    while num_tokens_from_messages(messages) >  4096: #8192
        #remove amount of text that is over the limit based on the proportion of tokens over the limit
        #remove from the end of the source, ensuring that there are enough tokens for a response (500 tokens)

        messages[1]["content"] = messages[1]["content"][:int(len(messages[1]["content"]) * (1 / over_prop))-800]

    print(Fore.RED + "Reader is working")
    response,a = generate_response_gpt3(messages)
    print(Fore.GREEN + response) #remove later on
    if not response:
        return json.dumps({})
    return response

def Writer(num):
    global outline
    #store writer.txt into content
    with open("writer.txt", "r") as f:
        writercontent = f.read()
    #store conversations/sources.txt into sources
    with open("conversations/sources.txt", "r") as f:
        sources = f.read()
    messages = [
        {"role":"system", "content":f'{writercontent}'},
        {"role":"user", "content":f'Outline: {outline}\n\n Sources: {sources} \n\n Section for you to write: {num}'}
        ]
    print(Fore.RED + "Writer is working")
    response,finish_reason = generate_response(messages)
    messages.append({"role":"assistant", "content":response})
    write_messages(messages, "writerlog")
    if finish_reason == 'length':
        extra_response,a = generate_response(messages)
        print(Fore.RED + "Writer is working")
        messages.append({"role":"assistant", "content":extra_response})
        write_messages(messages, "writerlog")
    
        response += " " + extra_response
    #create a file called essay.txt in the conversations folder
    with open(f"conversations/essay.txt", "a+") as f:
        f.write("\n" + "Section: " + str(num)  + "\n" + response)
        print(Fore.BLUE + "Saving to essay")

    return response

def Editor():
    global outline
    #store editor.txt into content
    with open("editor.txt", "r") as f:
        editorcontent = f.read()
    #store essay.txt into essay
    with open("conversations/essay.txt", "r") as f:
        essay = f.read()
    messages = [
        {"role":"system", "content":f'{editorcontent}'},
        {"role":"user", "content":f'Outline: {outline}\n\n Essay: {essay}'}
        ]
    #while "[STOP]" not in response:
    #print(Fore.RED + "Editor is working")
    response = ""
    while "[STOP]" not in response:
        print(Fore.RED + "Editor is working")
        response,a = generate_response(messages)
        if "[STOP]" in response:
            #remove [STOP] from anywhere in response
            response = response.replace("[STOP]", "")
            return response
        #get the first three characters of response
        first_three = response[:2]
        num = first_three[1]
        with open("essay.txt", "r") as f:
            essay = f.read()
        #get the index of the first $num$ in essay
        index = essay.find(first_three)
        #get the index of the next $num$ in essay
        next_index = essay.find("$" + str(int(num) + 1))
        #replace the text between the two $num$ with response
        essay = essay[:index] + response + essay[next_index:]
        messages.append({"role":"assistant", "content":response})
        messages.append({"role":"user", "content":f'Next work on section: {int(num) + 1}'})

    with open(f"\conversations\essay.txt", "w+") as f:
        f.write(response)
    return 

#def Citer():
    #return
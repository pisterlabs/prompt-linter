import datetime
import openai
import json
import requests
from PyPDF2 import PdfReader
from PIL import Image
import subprocess
import os
import time
import sys
from bs4 import BeautifulSoup
import re

global weatherAPIKey, ChatGPTModelToUse, ChatGPT4Model
ChatGPTModelToUse = "gpt-3.5-turbo" 
ChatGPT4Model = "gpt-4"

global CODESPATH
CODESPATH = '.codes'

with open('.keys.json', 'r') as f:
    params = json.load(f)
    openai.api_key = params['OPENAI_API_KEY']
    weatherAPIKey = params['weatherAPIKey']
    GOOGLE_API_KEY = params['GOOGLE_API_KEY']
    GOOGLE_SEARCH_ENGINE_ID = params['GOOGLE_SEARCH_ENGINE_ID']

JsonFNC = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country name, e.g. Istanbul, Turkey",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
    {
    "name": "get_current_time",
    "description": "Get the current time",
    "parameters": {
        "type": "object",
        "properties": {
        "location": {
            "type":"string",
            "description":"Location for time, e.g. Istanbul",
    },
        },
        "required" : [],
        }
    },
    {
    "name": "write_python_code_to_file",
    "description": """Write a python code to a file.
            Ensure the response can be parsed by Python json.loads.
            REMEMBER to format your response as JSON, using double quotes ("") around keys and string values, and commas (,) to separate items in arrays and objects. 
            IMPORTANTLY, to use a JSON object as a string in another JSON object, you need to escape the double quotes.
            """,
    "parameters": {
        "type": "object",
        "properties": {
        "code": {
            "type":"string",
            "description":"raw format of code to write directly to file",
    },
        "filename": {
            "type":"string",
            "description":"Name of the python file.",
    },
        },
        "required" : ["code","filename"],
        }
    },
    {
    "name": "write_to_file",
    "description": """Write a txt file.
            Ensure the response can be parsed by Python json.loads.
            REMEMBER to format your response as JSON, using double quotes ("") around keys and string values, and commas (,) to separate items in arrays and objects. 
            IMPORTANTLY, to use a JSON object as a string in another JSON object, you need to escape the double quotes.
    """,
    "parameters": {
        "type": "object",
        "properties": {
        "file_contents": {
            "type":"string",
            "description":"file contents",
    },
        "filename": {
            "type":"string",
            "description":"Name of the file to create. If none use a max 20 characters name to describe the txt file. ",
    },
        },
        "required" : ["code","filename"],
        }
    },
    {
    "name": "read_from_file",
    "description": "Read a file from disk and get contents",
    "parameters": {
        "type": "object",
        "properties": {
        "file_type": {
            "type":"string",
            "description":"file type like .txt or .csv.",
    },
        "filename": {
            "type":"string",
            "description":"Name of the file to read",
    },
        },
        "required" : ["filename","file_type"],
        }
    },
    {
    "name": "show_image",
    "description": "Read an image file from disk and show on screen",
    "parameters": {
        "type": "object",
        "properties": {
        "file_type": {
            "type":"string",
            "description":"file type like .jpg or .png",
    },
        "filename": {
            "type":"string",
            "description":"Name of the file to read",
    },
        },
        "required" : ["filename","file_type"],
        }
    },
    {
    "name": "search_in_google",
    "description": "Make a google search and get results from the first page",
    "parameters": {
        "type": "object",
        "properties": {
        "search_string": {
            "type":"string",
            "description":"The plain string to search in google. Used if a question asked that assistant cannot know because the question is about something in present time",
    },
        },
        "required" : ["search_string"],
        }
    },
    {
    "name": "browse_web_page",
    "description": "Browse a webpage and get its contents in plain text",
    "parameters": {
        "type": "object",
        "properties": {
        "full_url": {
            "type":"string",
            "description":"The full url of the webpage to get contents",
    },
        },
        "required" : ["full_url"],
        }
    },
    {
    "name": "run_python_code",
    "description": "Executes a python program (py file) and gets results",
    "parameters": {
        "type": "object",
        "properties": {
        "code_name": {
            "type":"string",
            "description":"name of the python file to run (include .py)",
    },
        "code_args": {
            "type":"string",
            "description":"arguments to pass to python program to run",
    },
        },
        "required" : ["code_name"],
        }
    }


]
# '','', 'browse_website', 'interrogate_image' 

def get_function_to_use(function_name):
    for function_data in JsonFNC:
        if function_data["name"] == function_name:
            return function_data
    return None  # Return None if no matching function_name is found

def RunGpt(myin,function_args):
    response = openai.ChatCompletion.create(
        model=ChatGPTModelToUse,  
        messages=myin,
        functions=function_args,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    updatetokens(response)
    return (response["choices"][0]["message"])

def search_in_google (myin,function_args):
    response_message = RunGpt(myin,function_args)
    if response_message.get("function_call"):
        funcName=response_message["function_call"]["name"]
        funcArgs= json.loads(response_message["function_call"]["arguments"])
        search_string = funcArgs.get("search_string")
        # Write me python code int the file 'ports.py' which checks if port 80 is in use
        SearchResults = search_in_google_DO(search_string)

        time.sleep(1)

        newq = myin
        newq.append ({"role": "assistant", "content": json.dumps(SearchResults)})
        # return(json.dumps(SearchResults))  #added for AGI
        newq.append ({"role": "user", "content": "Based on the google search, please give the answer"})
        HumanUnderstandable = convertGoogleToHuman (newq)
        return(HumanUnderstandable)
    
    else:
        return (response_message["content"])

def convertGoogleToHuman(myin):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=ChatGPTModelToUse,
                temperature = 0.3,
                messages=myin
            )
            updatetokens(response,'googleToHuman')
            response_message = response["choices"][0]["message"]
            return (response_message["content"])

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            exception_code = exc_obj.code
            print("Exception code:", exception_code)
            if exception_code == "context_length_exceeded":
                # print (f"OLD:\n{messages}")
                myin= myin[2:]
                # print (f"NEW:\n{messages}")
                print ("Decreasing size of chat..")
            else:
                # messages = messages[:-1]
                return ('ERROR CONNECTING TO Chat GPT\n')
        
def search_in_google_DO( query):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_SEARCH_ENGINE_ID}&q={query}"
        response = requests.get(url)
        data = response.json()
        results_small = []
        infoline={
            "Search Query": query,
            "Date and Time": get_current_time(),
            "Number of Results": data["queries"]['request'][0]['totalResults'],
            "Encoding": data["queries"]['request'][0]['outputEncoding']
        }
        results_small.append (infoline)
        if "items" in data:
            if "items" in data:
                for item in data["items"]:
                    result = {
                        "title": item["title"],
                        "link": item["link"],
                        "snippet": item["snippet"]
                    }
                    results_small.append(result)
            return(results_small)
        else:
            results_small.append ( {"SearchResult":"No Results Returned from Google"})
            return (results_small)
    except:
        results_small = []
        infoline={
            "Search Query": query,
            "Date and Time": datetime(),
            "Result":"Google Connect Error"
        }
        return(infoline)

def browse_web_page(myin,function_args ):
    response_message = RunGpt(myin,function_args)
    if response_message.get("function_call"):
        funcName=response_message["function_call"]["name"]
        funcArgs= json.loads(response_message["function_call"]["arguments"])
        web_url = funcArgs.get("full_url")
        # Write me python code int the file 'ports.py' which checks if port 80 is in use
        WebContents = browse_web_page_DO(web_url)

        time.sleep(1)

        newq = myin
        print (f'\033[91m{WebContents}\033[00m')
        newq.append ({"role": "assistant", "content": json.dumps(WebContents)})
        newq.append ({"role": "user", "content": "Based on the web page contents, what you say?"})

        HumanUnderstandable = convertGoogleToHuman (newq)
        return(HumanUnderstandable)
    
    else:
        return (response_message["content"])

def browse_web_page_DO(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        souptext = soup.get_text()
        souptext = re.sub('\n ','\n',souptext)
        souptext = re.sub(' \n','\n',souptext)
        for x in range(150):
            souptext2 = re.sub('\n\n','\n',souptext)
            souptext2 = re.sub('\t\t','\t',souptext)
            if souptext2 == souptext:
                break
            else:
                souptext = souptext2
        souptext2 = re.sub('\n','-',souptext2)
        souptext2 = re.sub('\t','-',souptext2)

        souptext2 = souptext2[0:2000]
        return (souptext2)
    except:
        return (f'The web page {url} failed to answer')
    
def write_to_file (myin,function_args):
    response_message = RunGpt(myin,function_args)
    if response_message.get("function_call"):
        funcName=response_message["function_call"]["name"]
        funcArgs= json.loads(response_message["function_call"]["arguments"])
        contents = funcArgs.get("file_contents")
        filename = funcArgs.get("filename")
        # Write me python code int the file 'ports.py' which checks if port 80 is in use
        answer = write_to_file_DO(filename,contents)
        return (f'\nThe written text is as follows: \n{str(contents)}\n\n{filename}\n')
    else:
        #messages.append(response_message) 
        return (response_message["content"])

def write_to_file_DO (filename,content):
    filename = convert_to_relative_path(filename)
    try:
        with open(f'{filename}', 'w',encoding='utf-8') as f:
            f.write(content)
        return (f'File written as {filename}.')
    except:
        return (f'Something went wrong on saving: {filename}.\n\n The file is like this: \n\n{content}\n')

def read_from_file (myin,function_args):
    response_message = RunGpt(myin,function_args)
    if response_message.get("function_call"):
        #funcName=response_message["function_call"]["name"]
        funcArgs= json.loads(response_message["function_call"]["arguments"])
        file_type = funcArgs.get("file_type")
        filename = funcArgs.get("filename")
        # Write me python code int the file 'ports.py' which checks if port 80 is in use
        return (read_from_file_DO(filename,file_type))
    else:
        #messages.append(response_message) 
        return (response_message["content"])

def read_from_file_DO (filename,file_type):
    if file_type in {'.txt','txt','text','json'}:
        try:
            with open(f'{CODESPATH}/{filename}', 'r',encoding='utf-8') as f:
                contents = f.read()
            return (f'Contents of the file are:\n\n{contents}.')
        except:
            return (f'Something went wrong on reading: {filename} Filetype:{file_type}.\n\n')
        
    elif file_type in {'.pdf','pdf','adobe pdf'}:
        try:
            reader = PdfReader(f'{CODESPATH}/{filename}')
            # read data from the file and put them into a variable called raw_text
            raw_text = ""
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text
            j=0
            while j<20:
                raw_text = raw_text.replace ('\n \n','\n')
                raw_text = raw_text.replace ('....','..')
                j += 1
            return (f'Contents of the PDF file are:\n\n{raw_text}.')
        except:
            return (f'Something went wrong on reading: {filename} Filetype:{file_type}.\n\n')
    else:
        return (f'Unknown Format: {filename} Filetype:{file_type}.\n\n')

def show_image (myin,function_args):
    response_message = RunGpt(myin,function_args)
    if response_message.get("function_call"):
        #funcName=response_message["function_call"]["name"]
        funcArgs= json.loads(response_message["function_call"]["arguments"])
        file_type = funcArgs.get("file_type")
        filename = funcArgs.get("filename")
        # Write me python code int the file 'ports.py' which checks if port 80 is in use
        return (show_image_DO(filename,file_type))
    else:
        #messages.append(response_message) 
        return (response_message["content"])

def show_image_DO (filename,file_type):
    if file_type in {'.jpg','.jpeg','jpg','jpeg','png','.png','tif','tiff','.tif','.tiff'}:
        try:
            image = Image.open(f'{CODESPATH}/{filename}')
            # Display the image
            image.show()
            return (f'image has been shown: {filename}')
        except:
            return (f'Something went wrong while showing: {filename} Filetype:{file_type}.\n\n')
    else:
        return (f'Cannot show {file_type} format: {filename} Filetype:{file_type}.\n\n')

def write_python_code_to_file(myin,function_args):
    response_message = RunGpt(myin,function_args)
    if response_message.get("function_call"):
        funcName=response_message["function_call"]["name"]
        try:
            funcArgs= json.loads(response_message["function_call"]["arguments"])
        except:  #need help in here, as sometimes the code is recieved as a text file, and I could not find a good way to format it to save as a code that is working.
            funcArgsA= json.dumps(response_message["function_call"]["arguments"])
            funcArgsA = funcArgsA.replace(r'\\n', '')
            funcArgsA = funcArgsA.replace(r'\\"', '"')
            funcArgs = json.loads(funcArgsA)

        try:
            code = funcArgs.get("code")
        except:
            code = str(funcArgs)

        filename = funcArgs.get("filename")
        # Write me python code int the file 'ports.py' which checks if port 80 is in use
        answer = write_python_code_to_file_DO(filename,code)
        return (f'\nThe Code is as follows: \n{code}\n\n{answer}\n')
    else:
        # messages.append(response_message) 
        return (response_message["content"])

def write_python_code_to_file_DO(filename,code):
    filename = convert_to_relative_path(filename)
    try:
        with open(f'{filename}', 'w') as f:
            f.write(code)
        return (f'👻 File written as {filename}.')
    except:
        return (f'Something went wrong on saving: {filename}.\n\n The Code is like this: \n\n{code}\n')

def get_weather(myin, function_args):
    response_message = RunGpt(myin,function_args)
    if response_message.get("function_call"):
        funcName=response_message["function_call"]["name"]
        funcArgs= json.loads(response_message["function_call"]["arguments"])
        location = funcArgs.get("location")
        unit = funcArgs.get("unit")
        # Write me python code int the file 'ports.py' which checks if port 80 is in use
        weather = get_weather_DO(location,unit)
        # print (weather)
        # messages.append ({"role": "assistant", "content": weather})
        return(weather)
    else:
        return (response_message["content"])

def get_weather_DO(location = 'Istanbul', unit="celcius"):
    api_key = weatherAPIKey
    base_url = "http://api.weatherapi.com/v1/current.json"
    ReplyText = get_weather_replies()

    # Create the parameters for the request
    params = {
        'key': api_key,
        'q': location
    }

    tempval = "Unable to get weather info.."

    # Make the request
    response = requests.get(base_url, params=params)

    # Parse the response
    if response.status_code == 200:
        data = response.json()
        
        weathertemp = data['current']['temp_c']
        weathercity = data['location']['name'] + ", " + data['location']['country']
        weatherstat =  data['current']['condition']['text']
        weatherwind =  data['current']['wind_kph']
        weatherfeel =  data['current']['feelslike_c']
        if unit == "fahrenheit":
            weathertemp = data['current']['temp_f']
            weatherfeel =  data['current']['feelslike_f']


        tempval = f"{ReplyText['curw']} {weathercity} \n\n"
        if unit == "fahrenheit":
            tempval = tempval + f"{ReplyText['temp']} {weathertemp}°F\n"
            tempval = tempval + (f"{ReplyText['feel']} {weatherfeel}°F\n")
        else:
            tempval = tempval + f"{ReplyText['temp']} {weathertemp}°C\n"
            tempval = tempval + (f"{ReplyText['feel']} {weatherfeel}°C\n")

        tempval = tempval + (f"{ReplyText['stat']} {weatherstat}\n")
        tempval = tempval + (f"{ReplyText['wind']} {weatherwind}{ReplyText['unit']} \n")
            
    return (tempval)

def get_weather_replies():
    return {
		"errCon": "Error getting weather info for",
		"errcity": "Error getting weather info. Please specify city",
		"errUnkn": "Error getting weather info (B)",
		"curw": "Current weather status in ",
		"temp": "Temperature: ",
		"feel": "Feels like : ",
		"stat": "Status     : ",
		"wind": "Wind       : ",
		"unit": " kmph"
    }

def get_current_time(myin='',function_args=''):
    nowtime = datetime.datetime.now()
    #print (nowtime)
    return (str(nowtime))

def run_python_code(myin,function_args):
    response_message = RunGpt(myin,function_args)
    if response_message.get("function_call"):
        funcName=response_message["function_call"]["name"]
        funcArgs= json.loads(response_message["function_call"]["arguments"])
        code_name = funcArgs.get("code_name")
        try:
            code_args = funcArgs.get("code_args").split()
        except:
            try:
                code_args = str(funcArgs.get("code_args")).split()
            except:
                code_args = funcArgs.get("code_args")

            
        # Write me python code int the file 'ports.py' which checks if port 80 is in use
        response2 = run_python_code_DO(code_name,code_args)
        return(response2)
    else:
        return (response_message["content"])

def run_python_code_DO(code_name,code_args="80"):

    try:
        result = subprocess.run(['python', f'{CODESPATH}\{code_name}'] + code_args, capture_output=True, text=True)
        output = result.stdout
        outerr= str(result.stderr)
        return_code = result.returncode

        print("Output:", output)
        print("Return code:", return_code)

        if return_code != 0:
            thisToReturn = f'Error. Return Code: {str(return_code)}\nError:{outerr[-200:]}'
        else:
            thisToReturn = f'The executed program {code_name} gave the result:\n{output}'
        return (thisToReturn)
    except Exception as e:
        return (str(e.args))

def updatetokens(response , callingfunction = 'function'):     # to save the usage of the OpenAI
    try:
        prompt_tokens=response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        total_tokens = response['usage']['total_tokens']
        tokens = [{"prompt":prompt_tokens, "completion":completion_tokens,"total":total_tokens, "function":callingfunction}]
        with open('tokens.json', 'r',encoding='utf-8') as openfile:
            oldtokens = json.load(openfile)
    except:
        oldtokens = []
    oldtokens.append(tokens)
    with open("tokens.json", "w",encoding='utf-8') as outfile:
        json.dump(oldtokens, outfile)

def convert_to_relative_path(absolute_path):    #HELPER to keep documents in subfolder
    # Get the current working directory
    base_path = os.getcwd()

    # Remove the drive letter from the absolute path
    drive, path = os.path.splitdrive(absolute_path)

    # Remove leading path separators
    path = path.lstrip(os.path.sep)

    # Append the remaining path after the current directory
    relative_path = os.path.join(CODESPATH, path)

    # Create the necessary directories
    directory = os.path.dirname(relative_path)
    os.makedirs(directory, exist_ok=True)

    return relative_path

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

if __name__ == '__main__':
    subprocess.run('venv/scripts/python AGI_Startup.py')
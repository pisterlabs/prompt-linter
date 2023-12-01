import glob
from collections import defaultdict
# ...[rest of your imports]...
import openai
import json
import os
import subprocess
import requests
import sys
from subprocess import check_output
import PySimpleGUI as sg
import pyperclip
from gpt_api_call import *
from prompts import *
from endpoints import *
import re
import requests
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
def count_tokens(text):
    return len(word_tokenize(text))
def popup_animation(path):
    return sg.popup_animated(path,message = None,background_color = None,text_color = None,font = None,no_titlebar = True,grab_anywhere = True,keep_on_top = True,location = (None, None),relative_location = (None, None),alpha_channel = None,time_between_frames = 0,transparent_color = None,title = "",icon = None)
def time_it():
    for i in range(1,10000):
        sg.one_line_progress_meter('My Meter', i+1, 10000, 'key','Optional message')
def copy_to_clip(text):
    pyperclip.copy(text)
def paste_from_clip():
    pyperclip.paste()
"copy input"
def get_response_types():
    return ['instruction', 'json', 'bash', 'text']
def predefined_command(user_input):
    command = ['net', 'user', '/domain', user_input]
    answer = check_output(command, args)
    decoded = answer.decode('utf8')
    return answer
def get_cmd_layout():
    return [
        [sg.Text('Enter a command to execute (e.g. dir or ls)')],[sg.Input(key='_IN_')],
        [sg.Output(size=(60,15))],[sg.Button('Run'), sg.Button('Exit')]
            ]
def runCommand(cmd, timeout=None, window=None):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ''
    for line in p.stdout:
        line = line.decode(errors='replace' if (sys.version_info) < (3, 5) else 'backslashreplace').rstrip()
        output += line
        print(line)
        window.Refresh() if window else None        # yes, a 1-line if, so shoot me
    retval = p.wait(timeout)
    return (retval, output)
def return_user_input(values):
    return values["-INPUT-"].strip()
def update_display(window,prompt):
    window["-RESPONSE_DISPLAY-"].update(prompt)
def win_update(win,st,upd):
    win[st].update(upd)
def win_read(win):
    return win.read()
def get_value(win,st):
    event, values = win_read(win)
    if st in values:
        return values[st]
    return ''
def ret_if(st,inp,if_it,end:str='',else_it=''):
    if inp != if_it:
        return st+inp+end
    return else_it
def create_pre_prompt(window):
    event, values = window.read()
    pre_prompt = create_prompt_keys(values)+ "\n " +ret_if('main_prompt: ',get_value(window,'-MAIN_PROMPT-'),'',end='\n',else_it='')+'\n'
    win_update(window,'-PRE_PROMPT_TOKEN_COUNT-',count_tokens(pre_prompt))
    prompt = pre_prompt+ get_content(values)
    update_display(window,prompt)
    return prompt
def get_content(values):
    tab_names,content = ["URL", "FILE", "IMAGE"],return_user_input(values)+'\n'
    for tab in tab_names:
        if values[f"-INCLUDE_{tab}-"]:  # Only include content if checkbox is checked
            content += values[f"-TAB_{tab}_CONTENT-"]
    return content
def calculate_max_chunk_size(user_input, num_chunks, model):
    max_tokens = get_default_tokens()
    prompt_template = "This is a sample prompt template with {} placeholders: {}, {}"#f"{user_input}, this is part {{}} of {num_chunks} for the data set {{}}:\n{{}}"
    # Estimate the number of tokens used by the chunk prompt by using a placeholder chunk and title
    prompt_tokens = count_tokens(prompt_template.format(1, "title", "chunk"))
    return max_tokens + prompt_tokens
def create_chunks(content, max_chunk_size):
    words = content.split()
    chunks = []
    current_chunk = []
    for word in words:
        if len(' '.join(current_chunk)) + len(word) > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
        current_chunk.append(word)
    chunks.append(' '.join(current_chunk))
    return chunks
def size_chunks(user_input,content,selected_model):
    initial_max_chunk_size = count_tokens(user_input)  # Or some other reasonable assumption
    # First, create 'content_chunks' using initial_max_chunk_size
    content_chunks = create_chunks(content, initial_max_chunk_size)
    # Then, calculate 'max_chunk_size' based on actual 'content_chunks'
    max_chunk_size = calculate_max_chunk_size(user_input, len(content_chunks), selected_model)
    # If max_chunk_size is different from initial_max_chunk_size, recreate 'content_chunks'
    if max_chunk_size != initial_max_chunk_size:
        content_chunks = create_chunks(content, max_chunk_size)
    return content_chunks
def return_chunk_prompt(k,content_chunks,data_set,chunk):
    return f",this is part {k + 1} of {len(content_chunks)} for the data set {data_set}:\n{chunk}"
def get_chunk_prompt(user_input,k,content_chunks,data_set,chunk):
    # Concatenate the user input and the selected tabs content
    return user_input + " " + return_chunk_prompt(k,content_chunks,data_set,chunk).strip()
def update_tokens(win):
    event, values = win_read(win)
    desired_perc = values['-DESIRED_COMPLETION_SELECT-']
    model_token_count = get_default_tokens()
    win_update(win,'-DESIRED_COMPLETION_CALCULATION-',int(model_token_count)*float(desired_perc)/float(100))
def get_list_Nums(i,k):
    ls=[]
    while i>0:
        ls.append(i)
        i-=k
    return ls
def get_pre_prompt():
    return [sg.Tab('pre-prompt',[[sg.Frame("Response Type:",[[sg.Combo(get_response_types(), default_value="instruction", key="-RESPONSE_TYPE-", enable_events=True)],])],
                                 [sg.T('Token Count:'),sg.Text(0,key="-RESPONSE_TOKEN_COUNT-"),sg.T('of '),sg.Text(get_default_tokens(),key="-MODEL_TOKEN_COUNT-"),
                                  sg.Frame('desired completion: ',[[sg.Combo(get_list_Nums(95,5),default_value=50,key='-DESIRED_COMPLETION_SELECT-',enable_events=True),
                                                                    sg.Text(get_default_tokens()*.5,key='-DESIRED_COMPLETION_CALCULATION-')]])],
                                 [sg.Multiline('', size=(None, None), key="-RESPONSE_DISPLAY-", autoscroll=True, expand_x=True, expand_y=True)]])]

def derive_values(values):
    return values["-INPUT-"].strip()+'\n'+create_prompt_keys(values),'','',values["-endpoint_var-"],values["-model_var-"]
def get_image_select():
    return [sg.Tab('Image Content',[[sg.Frame("image_select:",[[sg.FileBrowse("Select Image", key="-IMAGE_SELECT-", initial_folder="/home/bigrugz/Pictures", target="-IMAGE-", enable_events=True), sg.Button("Grab Image", key="-GRAB_IMAGE-", enable_events=True), sg.Checkbox("Include in prompt", key="-INCLUDE_IMAGE-", default=False)],[sg.Input('', key="-TAB_IMAGE_CONTENT-", enable_events=True, size=(25, 1), tooltip="Enter URL or select a IMAGE")]], expand_x=True, expand_y=True)],[sg.Image(data='', size=(None, None), key="-TAB_IMAGE_CONTENT-", expand_x=True, expand_y=True)]])]
def get_url_tab():
    return [sg.Tab('URL Content',[[sg.Frame("grab_url:",[[sg.Button("Select URL", key="-GRAB_URL-", enable_events=True), sg.Checkbox("Include in prompt", key="-INCLUDE_URL-", default=False)],[sg.Input('', key="-URL-", enable_events=True, size=(25, 1), tooltip="Enter URL or select a URL")]])],[sg.Multiline('', size=(None, None), key="-TAB_URL_CONTENT-", autoscroll=True, expand_x=True, expand_y=True)]])]
def get_pre_prompt():
    return [sg.Tab('pre-prompt',[[sg.Frame("Response Type:",[[sg.Combo(get_response_types(), default_value="instruction", key="-RESPONSE_TYPE-", enable_events=True)],])],
                                 [sg.Frame('Pre Prompt:',[[sg.Text(0,key='-PRE_PROMPT_TOKEN_COUNT-')]])],
                                 [sg.Frame('Token Count:',[[sg.Text(0,key="-RESPONSE_TOKEN_COUNT-")]])],
                                 [sg.Frame('Max Token:',[[sg.Text(get_default_tokens(),key="-MODEL_TOKEN_COUNT-")]])],
                                 [sg.Frame('Completion Percentage: ',[[sg.Combo(get_list_Nums(95,5),default_value=50,key='-DESIRED_COMPLETION_SELECT-',enable_events=True)]])],
                                 [sg.Frame('Completion Desired: ',[[sg.Text(get_default_tokens()*.5,key='-DESIRED_COMPLETION_CALCULATION-')]])],
                                 [sg.Frame('Completion Alloted: ',[[sg.Text(get_default_tokens(),key='-ALLOTED_COMPLETION_CALCULATION-')]])],
                                 [sg.Frame('input Total: ',[[sg.Text(get_default_tokens()-get_default_tokens()*.5,key='-INPUT_TOKEN_CALCULATION-')]])],
                                 [sg.Frame('input alloted: ',[[sg.Text(get_default_tokens()-get_default_tokens()*.5,key='-INPUT_TOKEN_ALLOTED-')]])],
                                 [sg.Multiline('', size=(None, None), key="-RESPONSE_DISPLAY-", autoscroll=True, expand_x=True, expand_y=True)]])]
def get_file_select():
    return [sg.Tab('File Content',[[sg.Frame("file_select:",[[sg.FileBrowse("Select File", key="-FILE_SELECT-", initial_folder="/home/bigrugz/Documents", target="-FILE-", enable_events=True), sg.Button("Grab File", key="-GRAB_FILE-", enable_events=True), sg.Checkbox("Include in prompt", key="-INCLUDE_FILE-", default=False)],[sg.Input('', key="-FILE-", enable_events=True, size=(25, 1), tooltip="Enter URL or select a FILE")]])],[sg.Multiline('', size=(None, None), key="-TAB_FILE_CONTENT-", autoscroll=True, expand_x=True, expand_y=True)]])]
def get_create_import_tab():
    return [sg.Tab('Create Import', [[sg.Text("Main Prompt")], [sg.Input(key="-MAIN_PROMPT-")],
                                     [sg.Button("Create Function", key="create_func")],
                                     [sg.Checkbox("get_json_response", key="get_json_response")],
                                     [sg.Checkbox("get_notation", key="get_notation")],
                                     [sg.Checkbox("get_title", key="get_title",default=True)],
                                      [sg.Checkbox("get_instruction", key="get_instruction")],
                                      [sg.Checkbox("get_inputs", key="get_inputs")],
                                      [sg.Checkbox("get_text", key="get_text")],
                                      [sg.Checkbox("get_bash", key="get_bash")],
                                      [sg.Checkbox('get_context', key='get_context')],
                                      [sg.Checkbox('get_security', key='get_security')],
                                      [sg.Checkbox('get_formatting', key='get_formatting')],
                                      [sg.Checkbox('get_validation', key='get_validation')],
                                      [sg.Checkbox('get_error_handling', key='get_error_handling')],
                                      [sg.Checkbox('get_revision', key='get_revision')],
                                      [sg.Checkbox('get_response', key='get_response')]])]

def get_gpt_layout():
    layout = [
        [[sg.Column([[sg.Multiline( key="-OUTPUT-", disabled=True, autoscroll=True,  expand_x=True, expand_y=True)],
                     [sg.Multiline( key="-INPUT-", autoscroll=True, expand_x=True, expand_y=True,enable_events=True)]], expand_x=True, expand_y=True),
          sg.Column([[sg.Pane([sg.Frame("",[[sg.Frame("response_inspection:",[[sg.TabGroup([get_pre_prompt(),get_url_tab(),
                                                                                              get_file_select(),
                                                                                              get_image_select(),
                                                                                              get_create_import_tab()],  # Add the new tab
                                                                                             size=(None, None), expand_x=True, expand_y=True)]],
                                                                        size=(None, None), expand_x=True, expand_y=True)]],
                               relief=sg.RELIEF_SUNKEN,expand_x=True,expand_y=True)],relief=sg.RELIEF_SUNKEN,show_handle=True,expand_x=True,expand_y=True),
                      ]], key="-SELECTIONS-", expand_x=True, expand_y=True)]],
            [sg.Frame("",[[sg.Button("Send",enable_events=True, bind_return_key=True),
                           sg.Button("copy_input",enable_events=True,bind_return_key=True),
                           sg.Button("EXAMPLE",enable_events=True, bind_return_key=True),
                           sg.Button("SELECT DIRECTORY", enable_events=True, bind_return_key=True),  # Add the new button here
                           sg.Text("Response Type:"),
                           sg.Combo(get_response_types(), default_value="instruction", key="-RESPONSE_TYPE-"),
                           sg.Text("Endpoint:"),
                           sg.Combo(list(endPoints().keys()), default_value=list(endPoints().keys())[0], key="-endpoint_var-",enable_events=True),
                           sg.Text("Model:"),
                           sg.Combo(get_models(list(endPoints().keys())[0]), default_value=get_models(list(endPoints().keys())[0])[0], key="-model_var-"),]],
                     relief=sg.RELIEF_SUNKEN,)]]
    window = sg.Window("ChatGPT Console", layout, resizable=True, size=(None, None), finalize=True)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        update_tokens(window)
        if event == "-endpoint_var-":
            window["-model_var-"].update(value=get_models(values["-endpoint_var-"])[0])
        if 'DESIRED_COMPLETION' in event:
            desired_perc = values['-DESIRED_COMPLETION_SELECT-']
            model_token_count = get_default_tokens()
            win_update(window,'-DESIRED_COMPLETION_CALCULATION-',int(model_token_count)*float(desired_perc)/float(100))
        if event == "copy_input":
            copy_to_clip(values["-INPUT-"])
        if event == "-INPUT-":
            prompt = create_pre_prompt(window)# In your event loop:
            prompt_count = count_tokens(prompt)
            win_update(window,"-RESPONSE_TOKEN_COUNT-",prompt_count)
            win_update(window,"-MODEL_TOKEN_COUNT-",get_default_tokens())
            win_update(window,"-DESIRED_COMPLETION_CALCULATION-",get_default_tokens()-prompt_count)
        if event == "Send" and values["-INPUT-"].strip():
            create_pre_prompt(window)
            js = {'model':values["-model_var-"],'prompt':'','max_tokens':get_default_tokens()}
            user_input,content,data_set,selected_endpoint,js['model'] = derive_values(values)
            # Get the content of the selected tabs
            content_chunks = size_chunks(user_input,get_content(values), js['model'])
            # Generate a separate prompt for each chunk
            for k, chunk in enumerate(content_chunks):
                js['prompt']=get_chunk_prompt(user_input,k,content_chunks,data_set,chunk)
                window["-OUTPUT-"].update(value=f"User: {js['prompt']}")
                message_tokens = sum([len(js['prompt']) for message in js['prompt']])
                js['max_tokens'] = get_default_tokens()-10 -(int(message_tokens)/2- int(check_token_size(create_prompt(js),int(message_tokens))))+int(message_tokens)/2
                input(js)
                response = raw_data(endpoint=selected_endpoint,js=js)
                #try_response(js,response)
                window["-OUTPUT-"].update(value=f"User: {js['prompt']}\n\n\n{js['model']}: {response}")
                # Clear the input
                window["-INPUT-"].update("")
        if event == "-endpoint_var-":
            window["-model_var-"].update(value=get_models()[values["-endpoint_var-"]][0])
        if event == "-URL-":
            if values["-URL-"]:
                display_image(window, values["-URL-"])
        if event == "-RESPONSE_TYPE-":
            window["-RESPONSE_DISPLAY-"].update(value=prePrompt(pre=str(values["-INPUT-"]),types=str(values["-RESPONSE_TYPE-"])))
        if "GRAB" in event:
            if event == "-GRAB_URL-":
                if values["-URL-"]:
                    window["-TAB_URL_CONTENT-"].update(read_url(values["-URL-"]))
            elif event == "-GRAB_FILE-":
                if values["-FILE-"]:
                    window["-TAB_FILE_CONTENT-"].update(read_file(values["-FILE-"]))
            elif event == "-GRAB_IMAGE-":
                if values["-IMAGE_SELECT-"]:
                    window["-TAB_IMAGE_CONTENT-"].update(read_image(values["-IMAGE_SELECT-"]))
                    
        # Add the new event case for the directory selection
        if event == "SELECT DIRECTORY":
          js = {'model':values["-model_var-"],'prompt':'','max_tokens':get_default_tokens()}
          dir_path = sg.popup_get_folder("Choose Directory", no_window=True)
          if dir_path:
                file_paths = glob.glob(dir_path + "/*.*")
                process_directory(file_paths, js, user_input, content, data_set, selected_endpoint)
                
        if event == "-FILE_SELECT-":
            if values["-FILE_SELECT-"]:
                window["-TAB_FILE_CONTENT-"].update(read_file(values["-FILE_SELECT-"]))
        if event == "-IMAGE_SELECT-":
            if values["-IMAGE_SELECT-"]:
                window["-TAB_IMAGE_CONTENT-"].update(read_image(values["-IMAGE_SELECT-"]))
                
    window.close()

# Define the new function `process_directory`
def process_directory(file_paths, js, user_input, content, data_set, selected_endpoint):
    dir_chunks = defaultdict(list)
    chunk_size = calculate_max_chunk_size(user_input, len(file_paths), js['model'])
    
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            file_content = file.read()
        
        # Create chunks for the file content
        content_chunks = create_chunks(file_content, chunk_size)
        
        # Store the chunks in the dictionary with the dataset key
        dir_chunks[data_set].extend(content_chunks)
    
    # Iterate over each data set and its chunks and send them to chatGPT
    for ds_key, ds_chunks in dir_chunks.items():
        for k, chunk in enumerate(ds_chunks):
            js['prompt'] = get_chunk_prompt(user_input, k, ds_chunks, ds_key, chunk)
            window["-OUTPUT-"].update(value=f"User: {js['prompt']}")
            
            message_tokens = sum([len(js['prompt']) for message in js['prompt']])
            js['max_tokens'] = get_default_tokens()-10 -(
                int(message_tokens)/2- int(check_token_size(create_prompt(js), int(message_tokens))))+int(message_tokens)/2
            response = raw_data(endpoint=selected_endpoint, js=js)
            window["-OUTPUT-"].update(value=f"User: {js['prompt']}\n\n\n{js['model']}: {response}")

if __name__ == "__main__":
    get_gpt_layout()

import tkinter as tk
import tkinter.font as tkfont
import colorsys
import os, sys, json, time, requests, ast, random, openai

import search_engines

from secret_things import openai_key

def col(ft, s):
    """For printing text with colors.
    
    Uses ansi escape sequences. (ft is "first two", s is "string")"""
    # black-30, red-31, green-32, yellow-33, blue-34, magenta-35, cyan-36, white-37
    u = '\u001b'
    numbers = dict([(string,30+n) for n, string in enumerate(('bl','re','gr','ye','blu','ma','cy','wh'))])
    n = numbers[ft]
    return f'{u}[{n}m{s}{u}[0m'
print('== api key check ==')
for apiname, obj in [
    ('openai', openai_key),
]:
    if obj == 'none':
        print(f'{col("re", "WARNING")} ~~ your {col("gr",apiname)} key in {col("cy","secret_things.py")} is "none", you need a proper one for the {col("gr",apiname)} api to work. ~~ {col("re","WARNING")}')
    else:
        print(col('gr', f'{apiname} key is not "none" (which is good)'))
print('== api key check ==')

def changecol(widget, value):
    # helper functions
    def int_to_hexadecimal(number):
        """Takes an integer between 0 and 255, returns the hexadecimal representation."""

        if number < 0 or number > 255:
            raise ValueError('must be between 0 and 255')

        digits = list("0123456789ABCDEF")
        first = number // 16
        second = number%16
        return ''.join(map(str,(digits[first],digits[second])))

    def hsv_to_hexcode(hsv, scale=1):
        """Takes a list of 3 numbers, returns hexcode.

        Divides each number by scale, multiplies by 255, rounds it, converts to 2-digit hex number

        Scale divides each number to make it a fraction.
            (So with scale=500, you can pass numbers between 0 and 500, instead of between 0 and 1.)
        """
        numbers = list(map(lambda n:n/scale, (hsv)))
        rgb = colorsys.hsv_to_rgb(*numbers)
        hexcode = '#' + ''.join(map(lambda n:int_to_hexadecimal(int(n*255)), rgb))
        return hexcode

    def get_hsv(color):
        rgb = tuple((c/65535 for c in widget.winfo_rgb(color)))
        hsv = colorsys.rgb_to_hsv(*rgb)
        return hsv

    print(f'value: "{value}"')
    parts = value.split(' ')
    print(f'parts: "{parts}"')
    if len(parts) == 1 and len(parts[0].split(',')) == 3:
        hsv = tuple(int(i)/10 for i in parts[0].split(','))
        print(f'hsv: "{hsv}"')
        hexcode = hsv_to_hexcode(hsv, scale=1)
        return hexcode
    if len(parts) != 2:
        print('invalid input, not 2 parts')
        return parts[0]
    
    color, addition = parts
    sign = addition[0]
    tup = tuple(addition[1:].split(','))
    print(f'tup: "{tup}"')
    if len(tup) != 3 or sign not in ['+', '-']:
        print('invalid input, not 3 numbers or no sign')
        return parts[0]
    if sign == '-':
        tup = tuple(-int(i) for i in tup)

    hsv = get_hsv(parts[0])
    new_hsv = [i+(int(j))/10 for i, j in zip(hsv, tup)]
    for n, i in enumerate(new_hsv):
        if i > 1:
            new_hsv[n] = 1
        elif i < 0:
            new_hsv[n] = 0

    print(f'new_hsv: {new_hsv}')
    new_hex = hsv_to_hexcode(new_hsv, scale=1)
    print(f'new_hex: {new_hex}')
    return new_hex

def set_tab_length(widget, length):
    font = tkfont.Font(font=widget['font'])
    tab_width = font.measure(' ' * 4)
    widget.config(tabs=(tab_width,))

def highlight_firstparts(widget):
    # start of word highlighting, inspired by https://twitter.com/InternetH0F/status/1656853851348008961

    # helper functions
    def int_to_hexadecimal(number):
        """Takes an integer between 0 and 255, returns the hexadecimal representation."""

        if number < 0 or number > 255:
            raise ValueError('must be between 0 and 255')

        digits = list("0123456789ABCDEF")
        first = number // 16
        second = number%16
        return ''.join(map(str,(digits[first],digits[second])))

    def hsv_to_hexcode(hsv, scale=1):
        """Takes a list of 3 numbers, returns hexcode.

        Divides each number by scale, multiplies by 255, rounds it, converts to 2-digit hex number

        Scale divides each number to make it a fraction.
            (So with scale=500, you can pass numbers between 0 and 500, instead of between 0 and 1.)
        """
        numbers = list(map(lambda n:n/scale, (hsv)))
        rgb = colorsys.hsv_to_rgb(*numbers)
        hexcode = '#' + ''.join(map(lambda n:int_to_hexadecimal(int(n*255)), rgb))
        return hexcode
    def convert_range(pair):
        """take normal range, return tkinter range"""
        assert len(pair) == 2
        assert len(pair[0]) == 2
        assert len(pair[1]) == 2
        def conv(tup):
            line, char = tup
            string = f'{line+1}.{char}'
            return string

        str1, str2 = map(conv, pair)
        tkinter_range = (str1, str2)

        return tkinter_range
    def get_hsv(color):
        rgb = tuple((c/65535 for c in widget.winfo_rgb(color)))
        hsv = colorsys.rgb_to_hsv(*rgb)
        return hsv
    def change_color(color, changers):
        # changers should be 3 callables, each taking a number between 0 and 1, and returning a number between 0 and 1
        # will be applied to hue/saturation/value, in that order.
        # (to make darker, reduce value)
        hsv = get_hsv(color)
        new_hsv = tuple(map(lambda n:changers[n](hsv[n]), range(3)))
        new_color = hsv_to_hexcode(new_hsv, scale=1)
        return new_color

    def get_changers():
        def third_fg_changer(n):
            # make darker
            n = max(0.1, n*0.7)
            return n
        fg_changers = [
            lambda n:n,
            lambda n:n,
            third_fg_changer,
        ]
        bg_changers = [
            lambda n:n,
            lambda n:n,
            lambda n:n,
        ]
        return fg_changers, bg_changers

    to_highlight = widget.get(1.0, 'end')[:-1]

    # get indices of words
    word_indices = []
    lines = to_highlight.split('\n')
    for line_n, line in enumerate(lines):
        idx = 0
        words = line.split(' ')
        for word in words:
            indices = ( (line_n,idx), (line_n,idx+len(word)) )
            word_indices.append(indices)
            idx += len(word) + 1  # +1 is for the space

    for pair in word_indices:
        ranges = convert_range(pair)
        widget.tag_add('wordstart', ranges[0], ranges[0]+' +2c')

    # keep bg the same, make fg darker.
    fg_changers, bg_changers = get_changers()
    new_fg = change_color(
        widget.cget('fg'),
        fg_changers
    )
    new_bg = change_color(
        widget.cget('bg'),
        bg_changers
    )
    settings = {
        'foreground': new_fg,
        'background': new_bg,
        'selectbackground': new_fg,
        'selectforeground': new_bg,
    }
    widget.tag_config('wordstart', **settings)

def get_paragraph(event):
    # returns the paragraph that the mouse is hovering over
    widget = event.widget
    x = widget.winfo_pointerx() - widget.winfo_rootx()
    y = widget.winfo_pointery() - widget.winfo_rooty()
    index = widget.index(f"@{x},{y}")
    return widget.get(f'{index} linestart', f'{index} lineend')

def get_selected(text_widget):
    # returns mouse-selected text from a tk.Text widget
    try:
        text_widget.selection_get()
    except:
        return None
    else:
        return text_widget.selection_get()

def make_json(dic, filename):
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(dic, f, indent=2)
        f.close()

def open_json(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        contents = json.load(f)
        f.close()
    return contents

def use_chatgpt(prompt, detailed_response=False):
    """Use the OpenAI chat API to get a response."""

    messages = [
        {
            'role':'user',
            'content':prompt,
        }
    ]

    folder_to_dump = 'chatgpt_responses'
    assert type(messages) is list
    for i in messages:
        assert type(i) is dict
        assert 'role' in i
        assert 'content' in i
        assert i['role'] in ['user', 'system', 'assistant']

    # create the request
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}"
    }
    data = {
        "model":"gpt-3.5-turbo",
        'messages':messages,
        'n':1,
        'temperature':0.5,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        print(vars(response))
        return "chatgpt error"
    response = response.json()

    folder_to_dump = 'backups'
    if folder_to_dump != None:
        # store the data and response for debugging
        if folder_to_dump not in os.listdir():
            os.mkdir(folder_to_dump)
        make_json({'data':data, 'response':response}, f'{folder_to_dump}/{time.time()}.json')

    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']
    total_tokens = response['usage']['total_tokens']
    ai_response = response['choices'][0]['message']['content']
    # parse and return ai response
    if detailed_response == True:
        return {
            'prompt_tokens':prompt_tokens,
            'completion_tokens':completion_tokens,
            'total_tokens':total_tokens,
            'ai_response':ai_response,
        }
    else:
        return ai_response

openai.api_key = openai_key
openai.organization = "org-jDwJqLdkah6kWhgpjCKXVcrD"
def use_chatgpt_stream(prompt, model, printfunc):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.5,
        stream=True
    )

    collection = []
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if delta == {}:
            break
        elif 'role' in delta:
            continue
        elif 'content' in delta:
            printfunc(delta['content'])
            collection.append(delta['content'])

    return ''.join(collection)

def change_fontsize(widget, increment):
    assert isinstance(increment, int)
    cur_font = widget['font']
    new_font = (cur_font[0], cur_font[1]+increment)

    widget['font'] = new_font

def between2(mess,start,end,strings_list=None):
    mess = mess.split(start)
    for i in range(len(mess)):
        mess[i] = mess[i].partition(end)[0]
    if strings_list == None:
        return mess[1:]
    else:
        filtered_mess = []
        for i in mess[1:]:
            for string in strings_list:
                if string in i:
                    filtered_mess.append(i)
        return filtered_mess
    mess = mess.split(start)
    for i in range(len(mess)):
        mess[i] = mess[i].partition(end)[0]
    return mess[1:]

def to_clipboard(text):
    pyperclip.copy(text)

def from_clipboard():
    spam = pyperclip.paste()
    return spam

def language_model(prompt, as_stream=False, print_func=print, model='gpt-3.5-turbo', editors=[], after_prompt=lambda:None):
    assert callable(after_prompt)

    def collect_replacers(editor, bracketed):
        def get_flagged(string, flag):
            if f'[{flag}]' in string and f'[/{flag}]' in string:
                flagged = string.partition(f'[{flag}]')[2].partition(f'[/{flag}]')[0][1:-1]
                return flagged
            else:
                return None

        d = {}
        for widget in editors:
            frame = widget.master
            if isinstance(frame, (tk.LabelFrame,)):
                title = frame['text']
            elif isinstance(frame, (tk.Toplevel,)):
                title = frame.title()
            else:
                continue
            full_text = widget.get('1.0', 'end-1c')

            for b in bracketed:
                if len(b.split('.')) != 2:
                    continue

                t, f = b.split('.')
                if t == title:
                    value = get_flagged(full_text, f)
                    if value == None:
                        print(f'could not find {b}')
                    else:
                        d[b] = value
        #d['clipboard'] = from_clipboard()  doesnt work
        return d

    recursion_count = 0
    while True:
        bracketed = between2(prompt, '{', '}')
        if len(bracketed) == 0:
            break
        replacers = collect_replacers(editors, bracketed)
        print(json.dumps(replacers, indent=4))
        for k,v in replacers.items():
            to_replace = '{'+k+'}'
            prompt = prompt.replace(to_replace, v)
        recursion_count += 1
        if recursion_count > 5:
            break

    print(col('cy','--- prompt below ---'))
    print(prompt)
    print(col('cy','--- prompt above ---'))

    # do backups and calculate costs
    if 'backups' not in os.listdir():
        os.mkdir('backups')
    if model == 'gpt-3.5-turbo' or model == 'gpt-3.5-turbo-16k':
        if as_stream:
            response = use_chatgpt_stream(prompt, model, print_func)
            return response
        else:
            response = use_chatgpt(prompt)
            print_func(response)
            return response

    # helper functions that are probably useless
    def response_to_dict(response_object):
        text = response_object.content.decode('utf-8')
        events = defaultdict(list)
        for event, data in re.findall(r"event:(\w+)\ndata:({.*})", text, re.MULTILINE):
            events[event].append(json.loads(data))
        result = dict(events)
        return result
    def dict_to_string(d):
        messages = []
        for item in d['completion']:
            messages.append(item['token'])
        return ''.join(messages)

    # prepare request
    url = "https://nat.dev/api/inference/text"
    payload_dict = {
        "prompt": prompt,
        "models": [
            {
                "name": f"openai:{model}",
                "tag": f"openai:{model}",
                "capabilities": ["chat"],
                "provider": "openai",
                "parameters": {
                    "temperature": 0.5,
                    "maximumLength": 400,
                    "topP": 1,
                    "presencePenalty": 0,
                    "frequencyPenalty": 0,
                    "stopSequences": [],
                    "numberOfSamples": 1,
                },
                "enabled": True,
                "selected": True,
            }
        ],
        "stream": True,
    }
    payload = json.dumps(payload_dict)
    session = """some stuff here"""
    headers = {
        "Content-Type": "text/plain;charset=UTF-8",
        "Accept": "*/*",
        "Referer": "https://nat.dev/",
        "Origin": "https://nat.dev",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Cookie": f"__session={session}",
    }
    if as_stream:
        response = requests.post(url, headers=headers, data=payload, stream=True)
    else:
        response = requests.post(url, headers=headers, data=payload)
    print(vars(response))
    # parse and return
    all_tokens = []
    if as_stream:
        for line in response.iter_lines():
            if line == b'event:status':
                continue
            else:
                data = str(line, 'utf-8').partition('data:')[2]
                if data == '':
                    continue
                token = json.loads(data)['token']
                if token == '[INITIALIZING]':
                    continue
                if token == '[COMPLETED]':
                    break
                else:
                    print_func(token)
                    all_tokens.append(token)
    else:
        d = response_to_dict(response)
        s = dict_to_string(d)
        print_func(s)

    with open(f'backups/{time.time()}.txt', 'w') as f:
        f.write(prompt + ''.join(all_tokens))
    
    def get_cost(model):
        costs = {
            'gpt-3.5-turbo': {
                'prompt': 0.0018,
                'completion': 0.0024,
            },
            'gpt-4': {
                'prompt': 0.036,
                'completion': 0.072,
            },
            'text-davinci-002':{
                'prompt': 0.024,
                'completion': 0.024,
            },
            'text-davinci-003':{
                'prompt': 0.024,
                'completion': 0.024,
            },
            'text-curie-001':{
                'prompt': 0.0024,
                'completion': 0.0024,
            },
        }
        prompt_tokens = len(prompt.split())
        completion_tokens = len(all_tokens)
        cost = costs[model]['prompt']*prompt_tokens + costs[model]['completion']*completion_tokens
        cost /= 1000
        return round(cost, 3)
    
    #print(f'spent: {col("gr",get_cost(model))} with {col("cy",model)}')
    print(f'with {col("cy",model)} spent: {col("cy",get_cost(model))}')
    print(f'with {col("ma","gpt-3.5-turbo")} wouldve spent: {col("ma",get_cost("gpt-3.5-turbo"))}')
    print(f'with {col("ma","gpt-4")} wouldve spent: {col("ma",get_cost("gpt-4"))}')

    #get_cost = lambda p_tok, c_tok: costs[model]['prompt']*p_tok + costs[model]['completion']*c_tok
    #cost = get_cost(len(prompt.split()), len(all_tokens))  # cost is per 1k tokens.
    #cost = cost/1000
    #print(f'cost: {cost}')
    #cents = round(cost*100, 2)
    #root.title(f'{cents} c')

    after_prompt()

    return ''.join(all_tokens)

def get_flagged(string, flag):
    if f'[{flag}]' in string and f'[/{flag}]' in string:
        flagged = string.partition(f'[{flag}]')[2].partition(f'[/{flag}]')[0][1:-1]
        return flagged
    else:
        return None

def get_flagged_multi(string, flag):
    _lst = between2(
        string,
        f'[{flag}]',
        f'[/{flag}]',
    )
    lst = [s[1:-1] for s in _lst]
    ret


def detect_flag(widget, idx):
    # search lines before and after idx, find flag, return flagged text

    after = widget.get(idx, tk.END)
    before = widget.get('1.0', idx)

    before_flag = None
    for line in reversed(before.split('\n')):
        if line.startswith('[ ') or line.endswith(' ]'):
            continue
        if line.startswith('[') and line.endswith(']'):
            before_flag = line[1:-1]
            break

    after_flag = None
    for line in after.split('\n'):
        if line.startswith('[ ') or line.endswith(' ]'):
            continue
        if line.startswith('[/') and line.endswith(']'):
            after_flag = line[2:-1]
            break

    if after_flag == before_flag:
        flagged = get_flagged(widget.get('1.0', tk.END), after_flag)
        return after_flag, flagged
    else:
        return None, None

def process_cmd(cmd):
    '''
    # replace if there is a macro
    editor_widget 
    macros_string = get_flagged(t.get(1.0, 'end')[:-1], 'macros')
    macros = {}
    for line in macros_string.split('\n'):
        if line == '':
            continue
        key, value = line.split(' --> ')
        macros[key] = value
    if cmd in macros:
        cmd = macros[cmd]
    with open('cmd_tracking.json', 'w') as f:
        json.dump(cmd_tracking, f, indent=2)

    print(f'processing cmd={cmd}')

    if cmd in cmd_tracking:
        elapsed = time.time() - cmd_tracking[cmd]
    else:
        elapsed = 0
    minutes = round(elapsed/60, 1)
    seconds = round(elapsed, 1)
    el = f'elapsed={minutes} minutes, {seconds} seconds'
    print(el)
    cmd_tracking[cmd] = time.time()
    root.title(el)'''

    # extract kwargs
    words = cmd.split(' ')
    if words[0] == 's':
        engine = words[1]
        if engine == 'google_site':
            engine = 'google site'
            site = words[2]
            term = ' '.join(words[3:])
            kwargs = {
                'engine':engine,
                'term':term,
                'site':site,
            }
        elif engine == 'google_img':
            engine = 'google img'
            term = ' '.join(words[2:])
            kwargs = {
                'engine':engine,
                'term':term,
            }
        else:
            engine = words[1]
            term = ' '.join(words[2:])
            kwargs = {
                'engine':engine,
                'term':term,
            }
        # use kwargs
        print(f'searching with kwargs={kwargs}')
        search_engines.do_search(**kwargs)
        res = json.dumps({'search type' if k == 'engine' else k:v for k,v in kwargs.items()}, indent=2)
    elif words[0] == 'dl':
        url = words[1]
        search_engines.youtubedl(url)
        res = str({
            'action':'download youtube video',
            'url':url,
        })
    elif words[0] == 't':
        url = words[1]
        search_engines.open_chrome_tab(url)
        res = str({
            'action':'open chrome tab',
            'url':url,
        })

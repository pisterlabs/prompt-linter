import openai, requests, json, os, time, sys
from secrets import openai_key

def text_append(path, appendage):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(appendage)

def text_create(path, content=''):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def text_read(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        contents = f.read()
    return contents

def make_json(dic, filename):
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(dic, f, indent=2)
        f.close()

def open_json(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        contents = json.load(f)
        f.close()
    return contents


def get_path_cli():
    """Commandline interaction to traverse folders and retrieve a file path."""

    def col(ft, s):
        """For printing text with colors.
        
        Uses ansi escape sequences. (ft is "first two", s is "string")"""
        # black-30, red-31, green-32, yellow-33, blue-34, magenta-35, cyan-36, white-37
        u = '\u001b'
        numbers = dict([(string,30+n) for n, string in enumerate(('bl','re','gr','ye','blu','ma','cy','wh'))])
        n = numbers[ft]
        return f'{u}[{n}m{s}{u}[0m'

    def get_type(path):
        if os.path.isdir(path):
            typ = 'dir'
        else:
            ext = path.split('.')[-1]
            typ = ext
        return typ

    def get_mapping(directory):
        assert os.path.isdir(directory)

        mapping = {}
        for f in os.listdir(directory):
            path = f'{directory}/{f}'
            typ = get_type(path)
            mapping[f] = typ
        return mapping

    type_to_col = {
        'py':'cy',
        'dir':'ye',
        'txt':'gr',
    }
    default_color = 'wh'

    current_dir = os.getcwd()
    while True:
        # show filenames with colors
        tups = list(get_mapping(current_dir).items())
        for n, tup in enumerate(tups):
            filename, typ = tup
            color = type_to_col.get(typ, default_color)
            print(n, col(color, filename))

        print('write a number to get the path, or go into the folder.')
        print()
        print(f'--- current_dir={col("ye",current_dir)} ---')
        print()
        n = input()
        if n == 'up':
            normal_path = os.path.abspath(current_dir)
            upper = normal_path.split('\\')
            current_dir = '\\'.join(upper[:-1])
            continue
        try:
            int(n)
        except:
            print(col('re', 'use an integer.'))
            exit()
        else:
            n = int(n)
        chosen_name, chosen_type = tups[n]
        if chosen_type == 'dir':
            current_dir += f'/{chosen_name}'
            continue
        else:
            path = f'{current_dir}/{chosen_name}'
            break

    return path

def col(ft, s):
    """For printing text with colors.
    
    Uses ansi escape sequences. (ft is "first two", s is "string")"""
    # black-30, red-31, green-32, yellow-33, blue-34, magenta-35, cyan-36, white-37
    u = '\u001b'
    numbers = dict([(string,30+n) for n, string in enumerate(('bl','re','gr','ye','blu','ma','cy','wh'))])
    n = numbers[ft]
    return f'{u}[{n}m{s}{u}[0m'

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_key}"
}

def print_colored(string, mapping):
    lines = []
    for line in string.split('\n'):
        for string, color in mapping.items():
            line = line.replace(string, col(color, string))
        lines.append(line)
    print('\n'.join(lines))

def get_commentary(path, n=3, temperature=0.5):

    mock = False  # for testing, so i dont have to wait for the api.
    if mock:
        fake_data = {
            'messages':[
                {
                    'role':'system',
                    'content':'your mom',
                },
                {
                    'role':'user',
                    'content':'yo',
                },
                {
                    'role':'assistant',
                    'content':'as a large language model, i cannot engage in interactions with or about, your mother.',
                },
            ]
        }
        fake_response = ''
        fake_output = 'raw'
        fake_ass = [f'fake resp {n}' for n in range(n)]
        fake_pretty = '\n'.join([f'{item}\n------\n' for item in fake_ass])
        return {
            'raw inputs':fake_data,
            'raw json':fake_output,
            'messages':fake_data['messages'],
            'assistant responses':fake_ass,
            'pretty print':fake_pretty,
        }

    
    path = path.replace('/', '\\')
    path = path.replace('\\', '/')
    filename = path.split('/')[-1]

    code = text_read(path)
    system_message = '''
Your job is to be my assistant that is always watching over my shoulder as I write Python code.
I will show you the code of the program I am in the process of currently writing, and I would like you to give me 3 things: a summary of functions/classes, commentary, and something that is interesting or strange about it.

This is what your response should generally look like:

Summary of functions and classes:
- do_thing(obj) -- this function does a thing, takes 1 argument
- Car -- this class stores cars
- etc..

Commentary:
The intent seems to be to do things with car data.
Although, (blabla)....
But also (blabla) ...

Interesting or strange:
I am not sure, but...
'''[1:-1]
    user_part = 'so.. heres the code im writing right now, i want your analysis.'
    if filename != None:
        user_part += '\n\n' + filename
    code_part = f'```python\n{code}\n```'
    data = {
        "model":"gpt-3.5-turbo",
        'messages':[
            {
                'role':'system',
                'content':system_message,
            },
            {
                'role':'user',
                'content': user_part + '\n\n' + code_part,
            }
        ],
        'n':n,
        'temperature':temperature,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    raw_json = response.json()
    print(raw_json)
    assistant_responses = [choice['message']['content'] for choice in raw_json['choices']]

    pp_lines = []
    def add_pp(ft, string):
        pp_lines.append(col(ft, string))
    add_pp('ma', system_message)
    add_pp('gr', user_part)
    add_pp('cy', code_part)
    for rep in assistant_responses:
        for paragraph in rep.split('\n\n'):
            # first line is white, rest is yella.
            par_lines = paragraph.split('\n')
            if par_lines[0][-1] == ':':
                add_pp('wh', par_lines[0])
            else:
                add_pp('ye', par_lines[0])
            add_pp('ye', '\n'.join(par_lines[1:]))
            add_pp('wh', '')

        add_pp('ma', '-_'*20)
    pretty_print = '\n'.join(pp_lines)

    return {
        'raw inputs':data,
        'raw json':raw_json,
        'messages':data['messages'],
        'code':code,
        'assistant responses':assistant_responses,
        'pretty print':pretty_print,
    }

# for additional interactions with turbo-kun.
def talk_again(messages_lod, n=3, temperature=0.5):
    data = {
        "model":"gpt-3.5-turbo",
        'messages':messages_lod,
        'n':n,
        'temperature':temperature,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    raw_json = response.json()
    assistant_responses = [choice['message']['content'] for choice in raw_json['choices']]

    return {
        'raw input':data,
        'raw json':raw_json,
        'assistant responses':assistant_responses,
        'messages':data['messages'],
    }

inp = sys.argv
if len(inp) == 2:
    path = inp[1]
    args = [path]
elif len(inp) == 3:
    path = inp[1]
    n = int(inp[2])
    args = [path, n]
elif len(inp) == 4:
    path = inp[1]
    n = int(inp[2])
    temp = float(inp[3])
    args = [path, n, temp]
else:
    #print(col('re','wrong args. pass a path followed by (optional) a small integer and a float between 0.0 and 1.5'))
    #exit()
    print(col('gr', 'wrong args. lets find the path.'))
    while True:
        path = get_path_cli()
        
        color_mapping = {
            'def':'blu',
            'return ':'ma',
            'print(':'ye',
            'len(':'ye',
            'int(':'ye',
            'class ':'cy',
            '=':'gr',
            'while ':'ma',
            'for ':'ma',
            'range(':'ye',
            "'":'re',
            '"':'re',
        }
        print(col('gr', '='*20))
        print(col('gr', '='*20))
        print(col('gr', '='*20))
        print_colored(text_read(path), color_mapping)  # prints code with some text colored as instructed by color_mapping
        args = [path, 3, 0.8]
        print(f'args:{args}')
        i = input('continue? (yes/no)\n')
        if i == 'yes':
            print('calling api...')
            break
        elif i == 'no':
            print('going again.')
        else:
            print(col('re', 'bruh i said "yes or no"'))
        
ext = path.split('.')[-1]

if ext == 'py':
    commentary = get_commentary(*args)
    # for checking things.
    print(col('blu', 'raw inputs'))
    print(json.dumps(commentary['raw inputs'], indent=2))
    print(col('blu', 'raw output json'))
    print(json.dumps(commentary['raw json'], indent=2))
    if 'pretty print' in commentary:
        print(commentary['pretty print'])
    else:
        m = commentary['messages']
        ass_reps = commentary['assistant responses']
        print(col('ma', m[0]['content']))  # initial system message
        print(col('ye', m[1]['content']))  # the user message
        for rep in ass_reps:
            print(col('cy', rep))
            print(col('gr', '-_'*20))

    # continue with additional interactions, using def talk_again
    current_messages = commentary['messages']
    assistant_responses = commentary['assistant responses']
    while True:
        i = input(col('gr','If you want to talk again, write the number of the message to respond to.\n'))
        print('-')
        try:
            assistant_responses[int(i)]
        except:
            print(col('re', 'wrong nombah init'))
            continue
        else:
            chosen_version = assistant_responses[int(i)]
            print(f'you chose:\n{chosen_version}')
        i = input(col('gr','Write your response.\n'))
        print('-')
        for tup in [('assistant',chosen_version), ('user',i)]:
            current_messages.append({'role':tup[0], 'content':tup[1]})
        
        returned_dict = talk_again(current_messages)  # call api.

        assistant_responses = returned_dict['assistant responses']

        # print stuff
        ## for diagnostics.
        print('raw input', returned_dict['raw input'])
        print('raw output', returned_dict['raw json'])

        ## showing gpt responses.
        for response in assistant_responses:
            print(col('gr', '-_'*20))
            print(col('ye', response))

        # prepare for next iteration of loop
        current_messages = returned_dict['messages']
else:
    print('extension must be python')

import logging
from time import sleep
import os
import openai
from .prompts import PROMPT_START, PROMPT_HEADER, prompts


KEY = os.getenv('OPENAI_API_KEY')

if KEY is None:
    logging.error('Can not find API key; please set the "OPENAI_API_KEY" environment variable and restart.')
    breakpoint()

openai.api_key = KEY
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


def split_line_tokens(line):
    return [item.strip().lower() for item in line.split(' ')]


def split_file(filetxt):
    """
    splits text with multiple IDL procedures/functions
    into a list of individual procedures/functions

    kludgy, but works most of the time
    """
    if 'end\n' not in filetxt.lower():
        return [filetxt]
    out = []
    filelines = filetxt.split('\n')
    for line in filelines:
        if line.lower().strip() == 'end':
            out.append('end')
        else:
            out.append(line)

    items = []
    current = []
    # keep track of case statements, since those can have end's in them
    case = False

    for line in out:
        current.append(line)
        tokens = split_line_tokens(line)
        if 'case' in tokens:
            case = True
        if 'endcase' in tokens:
            case = False
        if 'end' in tokens:
            if not case:
                # should be the end of the function
                items.append('\n'.join(current))
                current = []
    return items


def estimate_tokens(text):
    """
    estimates the number of tokens in some text

    this estimate is from:
        https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
    """
    return len(text) / 4.0


def get_prompt(text, header=False):
    """
    generates the input to send to the model
    """
    if header:
        out = PROMPT_START + PROMPT_HEADER
    else:
        out = PROMPT_START
    for prompt in prompts.keys():
        if prompt in text.lower():
            out = out + prompts[prompt]
    out = out + '\nIDL:\n' + text + '\n' + """\nPython:\n"""
    return out


def convert(file,
            raw=False,
            max_tokens=4096,
            best_of=5,
            header=True,
            dynamic_tokens=True,
            token_padding=0.61,
            tmp_output=False):
    """

    """
    if not raw:
        directory, filename = os.path.split(file)
        out_file = filename[:-4] + '.py'
        filetxt = open(file, 'r').read()
    else:
        filetxt = file

    if header:
        logging.info('including header example...')

    items = split_file(filetxt)

    logging.info('number of functions/procedures: ' + str(len(items)))
    out = ''

    for index, item in enumerate(items):
        logging.info('item #' + str(index + 1))
        # prompt = get_prompt(item, header=header)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert scientific programmer, with detailed knowledge of Interactive Data Language (IDL) and Python. Your sole purpose in life is to convert IDL code to Python code. The user provides some IDL and Python examples, followed by some IDL code, and you return the Python code."},
                {"role": "user", "content": f"The code is:\n{item}"},
            ]
        )

        output = response['choices'][0]['message']['content']
        full_output = output

        if tmp_output and not raw:
            # useful for debugging strange output
            with open(os.path.join(directory, filename[:-4] + '_tmp_full_' + str(index) + '.py'), 'w') as f:
                f.write(full_output)
            with open(os.path.join(directory, filename[:-4] + '_tmp_' + str(index) + '.py'), 'w') as f:
                f.write(output)
            # with open(os.path.join(directory, filename[:-4] + '_prompt_' + str(index) + '.pro'), 'w') as f:
            #     f.write(prompt)

        out = out + '\n\n' + output

    if raw:
        return out

    with open(os.path.join(directory, out_file), 'w') as f:
        f.write(out)

    logging.info('done!')
    logging.info('output: ' + os.path.join(directory, out_file))


def convert_folders(folder=None, max_tokens=4096, skip_existing=True, header=True):
    """
    convert all folders and subfolders
    """
    if folder is None:
        logging.error('No folder specified.')
        return

    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in [f for f in filenames if f.endswith(".pro")]:
            if skip_existing:
                if os.path.isfile(os.path.join(dirpath, filename[:-4] + '.py')):
                    continue
            logging.info('processing: ' + filename)
            try:
                convert(os.path.join(dirpath, filename), max_tokens=max_tokens, header=header)
            except:
                logging.error('problem with: ' + os.path.join(dirpath, filename))
                continue
            # sleep for 30 seconds to avoid hitting the API's rate limits
            sleep(30)



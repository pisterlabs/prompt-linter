from __future__ import with_statement

import os.path, sys
import pathlib
import re
sys.path += [
    os.path.dirname(__file__),
    os.path.join(os.path.dirname(__file__), 'third_party.zip'),
    os.path.join(os.path.dirname(__file__), f'third_party_{os.name}.zip'),
]


import wingapi
import wingutils
import guiutils
import shared


GPT_3_5_TURBO = 'gpt-3.5-turbo'
GPT_4 = 'gpt-4'
GPT_4_TURBO = 'gpt-4-1106-preview'

SELECTED_MODEL = GPT_4_TURBO

TOP_WHITESPACE_PATTERN = re.compile('^(\s*)')
BOTTOM_WHITESPACE_PATTERN = re.compile('(\s*)$')



TEMPLATE = '''
This automated call is being made by a plugin to Wing IDE that allows you to help the user with their code. Therefore your reply must not contain any preambles or other explanations, because it'll make the plugin code break.

The user currently has the file `{file_name}` open.

This is the user's message / request (`user_message`):

```
{user_message}
```


Below is the code in the currently open file (`entire_file_content`):

```
{entire_file_content}
```

Below is the code that's selected (`selected_code`). The user's message / request pertains only for this segment of code, and it's the only segment you're allowed to change.

```
{selected_code}
```

Your reply should be a single code block with no explanations. We'll perform `new_code = entire_file_content.replace(selected_code, your_reply)`, which means that the selected code will be deleted and replaced with your code.
'''

pattern = re.compile(r'(?s)```(?:[a-z0-9]*)\n(.+)\n```')



def _ask_chatgpt(message: str) -> str:
    import openai
    openai.api_key = (pathlib.Path.home() / '.roberto-openai-key').read_text()
    completion = openai.ChatCompletion.create(model=SELECTED_MODEL,
                                              messages=[{'role': 'user', 'content': message}])
    return completion['choices'][0]['message']['content']


def _fix_code_with_chatgpt(file_name: str, entire_file_content: str, selected_code: str,
                          user_message: str) -> str:
    assert entire_file_content.count(selected_code) == 1
    populated_template = TEMPLATE.format(**locals())
    top_whitespace = TOP_WHITESPACE_PATTERN.search(selected_code).group(0)
    bottom_whitespace = BOTTOM_WHITESPACE_PATTERN.search(selected_code).group(0)
    response = _ask_chatgpt(populated_template)
    result = pattern.search(response).group(1)
    result = TOP_WHITESPACE_PATTERN.sub(top_whitespace, result)
    result = BOTTOM_WHITESPACE_PATTERN.sub(bottom_whitespace, result)
    return result


def roberto(user_message):
    '''
    Use ChatGPT to modify your code.

    Put your OpenAI API key in `~/.roberto-openai-key` first. Mark some segment of code and invoke
    `roberto`. Then in the dialog, write in text what changes you want ChatGPT to make. Then wait,
    sometimes even several minutes, for it to complete.

    Suggested key combination: `Insert Ctrl-Alt-R`
    '''
    editor = wingapi.gApplication.GetActiveEditor()
    assert isinstance(editor, wingapi.CAPIEditor)
    document = editor.GetDocument()
    assert isinstance(document, wingapi.CAPIDocument)
    start, end = selection = editor.GetSelection()

    new_code = _fix_code_with_chatgpt(document.GetFilename(),
                                      document.GetCharRange(0, document.GetLength()),
                                      document.GetCharRange(*selection),
                                      user_message)

    with shared.UndoableAction(document):

        document.DeleteChars(start, end-1)
        document.InsertChars(start, new_code)
        editor.SetSelection(start,
                            start + len(new_code))

roberto.arginfo = {
    'user_message': wingapi.CArgInfo(
        'The message to send to ChatGPT',
        wingutils.datatype.CType(''),
        guiutils.formbuilder.CSmallTextGui(),
        'User message:'
    ),
}
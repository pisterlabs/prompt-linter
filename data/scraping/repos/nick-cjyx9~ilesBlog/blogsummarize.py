# 使用者必读： 由于max_token的限制，我将字数限制到3600字内，也就是3600字以外的文章不会被摘要！
import openai as o
from os import getenv, listdir, path
import re

o.api_base = 'https://p0.kamiya.dev/api/openai'
o.api_key = getenv('KAMIYA_API_KEY')
# o.api_key = getenv('OPENAI_API_KEY')
def get_completion(messages, model="gpt-3.5-turbo", temperature=0.3):
    response = o.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]

# {'file':(none|'abstract'), }
with open('E:/dev/RinaBlog/misc/scripts/data.json', 'r', encoding='utf-8') as f:
    try:
        data = eval(f.read())
    except:
        data = {}
    f.close()

messages = [
    {
        'role':'user', 
        'content':'给出下列博文的完整摘要：'
    },
    {
        'role':'user',
        'content': ''
    }
]

def convert_markdown_to_text(md, options=None):
    options = options or {}
    options.setdefault('listUnicodeChar', False)
    options.setdefault('stripListLeaders', True)
    options.setdefault('gfm', True)
    options.setdefault('useImgAltText', True)
    options.setdefault('abbr', False)
    options.setdefault('replaceLinksWithURL', False)
    options.setdefault('htmlTagsToSkip', [])
    output = md or ''
    pattern = r'<BlockSummary>.*?</BlockSummary>'
    output = re.sub(pattern, '', output, flags=re.DOTALL)
    # Remove frontmatter
    pattern = r'^---\n.*?\n---\n'
    output = re.sub(pattern, '', output, flags=re.MULTILINE | re.DOTALL, count=1)
    # Remove horizontal rules
    output = re.sub(
        r'^(-\s*?|\*\s*?|_\s*?){3,}\s*', '', output, flags=re.MULTILINE)
    try:
        if options['stripListLeaders']:
            list_regex = r'^([\s\t]*)([\*\-\+]|\d+\.)\s+'
            if options['listUnicodeChar']:
                output = re.sub(
                    list_regex, options['listUnicodeChar'] + ' \\1', output, flags=re.MULTILINE)
            else:
                output = re.sub(list_regex, '\\1', output, flags=re.MULTILINE)
        if options['gfm']:
            output = re.sub(r'\n={2,}\n', '\n', output)  # Header
            output = re.sub(r'~{3}.*\n', '', output)  # Fenced codeblocks
            output = re.sub(r'~~', '', output)  # Strikethrough
            output = re.sub(r'`{3}.*\n', '', output)  # Fenced codeblocks
        if options['abbr']:
            # Remove abbreviations
            output = re.sub(r'\*\[.*\]:.*\n', '', output)
        output = re.sub(r'<[^>]*>', '', output)  # Remove HTML tags
        html_tags_to_skip = '(?!{})'.format(
            '|'.join(options['htmlTagsToSkip']))
        html_replace_regex = r'<{}[^>]*>'.format(html_tags_to_skip)
        output = re.sub(html_replace_regex, '', output, flags=re.IGNORECASE)

        # Remove setext-style headers
        output = re.sub(r'^[=\-]{2,}\s*$', '', output, flags=re.MULTILINE)
        output = re.sub(r'\[\^.+?\](\: .*?$)?', '', output)  # Remove footnotes
        # Remove reference-style links
        output = re.sub(r'\s{0,2}\[.*?\]: .*?$', '', output)
        output = re.sub(r'\!\[(.*?)\][\[\(].*?[\]\)]', lambda m: m.group(1)
                        if options['useImgAltText'] else '', output)  # Remove images
        output = re.sub(r'\[([^\]]*?)\][\[\(].*?[\]\)]', lambda m: m.group(
            2) if options['replaceLinksWithURL'] else m.group(1), output)  # Remove inline links
        output = re.sub(r'^(\n)?\s{0,3}>\s?', lambda m: m.group(1) if m.group(
            1) else '', output, flags=re.MULTILINE)  # Remove blockquotes
        # Remove reference-style links
        output = re.sub(
            r'^\s{1,2}\[(.*?)\]: (\S+)( ".*?")?\s*$', '', output, flags=re.MULTILINE)
        output = re.sub(r'^(#+)\s*(.*?)\s*$', r'\2', output, flags=re.MULTILINE)  # Remove ATX-style headers
        output = re.sub(r'(\*+)([^*]+)\1', r'\2', output)  # Remove * emphasis
        output = re.sub(r'(^|\W)(_+)(\S.*?\S)?\2($|\W)', r'\1\3\4', output)  # Remove _ emphasis
        output = re.sub(r'(`{3,})(.*?)\1', r'\2', output,
                        flags=re.DOTALL)  # Remove code blocks
        output = re.sub(r'`(.+?)`', r'\1', output)  # Remove inline code
        output = re.sub(r'~(.*?)~', r'\1', output)  # Replace strike through
        pattern = r'\n\s*\n'
        output = re.sub(pattern, '\n', output)
    except Exception as e:
        print(e)
        return output
    return output


path = 'E:/dev/RinaBlog/src/pages/post/'
files = listdir(path)
for file in files:
    if('.mdx' not in file):
        continue
    print('START:::', file, end=' ')
    if file not in data.keys():
        data[file] = ''
    if data[file] == '' or data[file] == None:
        with open(path+file, 'r', encoding='utf-8') as f:
            item = f.read()
            if(('draft: true' in item) or ('draft:true' in item)): 
                print('Is Draft. Continue.')
                continue
            item = convert_markdown_to_text(item)[:3601].replace('\n','\\n')
            f.close()
        messages[1]['content'] = item
        print('Sending Message to ChatGPT!',end=' ')
        try:
            response = get_completion(messages)
        except Exception as e:
            response = ''
            print('ERR:', e)
        data[file] = response
        print('Done.')
        print(response)
    else:
        print('Already Generated!')

with open('E:/dev/RinaBlog/misc/scripts/data.json', 'w', encoding='utf-8') as f:
    processed = str(data).replace("'",'"')
    f.write(processed)
    f.close()
with open('E:\dev\RinaBlog\modules\AIabstract.ts','r',encoding='utf-8') as f:
    origin_code = f.read()
    f.close()
cp = list(origin_code)
cp[origin_code.find('// :::DATASTART:::')+19:origin_code.find('// :::DATAEND:::')-1]='    var data = '+processed+';\n'
foo = ''
for i in cp:
    foo+=i
# print(foo)
with open('E:\dev\RinaBlog\modules\AIabstract.ts','w',encoding='utf-8') as f:
    f.write(foo)
    f.close()

input('Saved!Press any key to exit!')

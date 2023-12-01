import openai
from fire import Fire
from time import sleep
from pathlib import Path
import shutil
from tqdm import tqdm

def makeprompt(pycode:str):
    prompt = f'''As a fluent programmer to both python and kotlin, would you help me to port the following python code into a kotlin code that exactly do the same thing?  Don't forget to name is_spam into isSpam when porting it.
    
```python
{pycode}
```
'''
    return prompt

def is_output_usable(chatgpt_out:str):
    codeheader = '```kotlin'
    triple = '```'
    if codeheader not in chatgpt_out:
        return False
    if triple not in chatgpt_out:
        return False
    codeonly = chatgpt_out.split('```kotlin\n')[1].replace('```','')
    if not 'isSpam' in codeonly:
        return False
    if len(codeonly) <= 1:
        return False
    return True

def postprocess(chatgpt_out:str):
    return chatgpt_out.split('```kotlin\n')[1].split('```')[0]


def do_code_gpt(prompt:str, model:str='gpt-3.5-turbo')->str:
    while True:
        try: 
            response = openai.ChatCompletion.create(
                    model = model,
                    messages = [
                        dict(role='system', content='You are a helpful assistant.'),
                        dict(role='user', content=prompt)
                        ]
                )
            break
        except:
            print(f'{model} api failed, retrying in 1 seconds')
            sleep(1)
            continue
    content = response['choices'][0]['message']['content']
    return content

def main(dbg=False):
    openai.api_key = open('apikey.txt').readlines()[0].strip()
    
    # make
    trans_dir = Path('wrapped_kotlin_ported/')
    if trans_dir.exists():
        shutil.rmtree(trans_dir)
    trans_dir_full = trans_dir/'funcs'
    trans_dir_full.mkdir(parents=True)
    print('cp 5_1_main_kt_by_chatgpt.kt wrapped_kotlin_ported/')
    shutil.copy('5_1_main_kt_by_chatgpt.kt', 'wrapped_kotlin_ported/')
    print('cp 3_inputmsgs.csv wrapped_kotlin_ported/')
    shutil.copy('3_inputmsgs.csv', 'wrapped_kotlin_ported/')
    
    
 
    paths = Path('wrapped/').glob("funcs/*.py")

    for p in tqdm(list(paths)):
        pycode = "".join(open(p).readlines())
        prompt = makeprompt(pycode)
        count=0
        while True:
            chatgptout = do_code_gpt(prompt)
            ok = is_output_usable(chatgptout)
            if ok:
                break
            count+=1
            print(f"{str(p)}")
            print(f'\tgptoutput not usable. trying again ({count=})')
        kotlincode = postprocess(chatgptout)
        newpath = str(p).replace('wrapped/', 'wrapped_kotlin_ported/').replace('.py', '.kt')
        with open(newpath, 'w') as f:
            f.write(kotlincode)
            print(f"writing to \n\t{str(newpath)}")
        
        






    

if __name__=='__main__':
    Fire(main)
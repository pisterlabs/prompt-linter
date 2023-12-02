import openai
import time
import os

API_KEY = YOUR_KEY

def query_openai_model(api_key: str, prompt: str, model: str = "gpt-3.5-turbo-0301", max_tokens: int = 128, temperature: float = 0):
    openai.api_key = api_key

    completions = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    output = completions.choices[0].message.content.strip()
    return output


def infer(infile, outfile):
    api_key = API_KEY
    
    existing_outputs = []
    if os.path.exists(outfile):
        with open(outfile, 'r') as fin:
            for line in fin:
                existing_outputs.append(line)
    

    with open(infile, 'r') as fin,\
        open(outfile, 'w') as fout:
        count = 0
        for line in fin:
            prompts = eval(line.strip('\n'))
            if count < len(existing_outputs):
                outputs = eval(existing_outputs[count].strip('\n'))
                for key in prompts:
                    #if key not in outputs:
                    if key in ['zs_resp', 'fs_resp']:
                        prompt = prompts[key]
                        flag = True
                        while flag:
                            try:
                                if key in ['zs_resp', 'fs_resp']:
                                    output = query_openai_model(api_key, prompt,max_tokens=40)
                                elif key in ['zs', 'zs-pcot']:
                                    output = query_openai_model(api_key, prompt,max_tokens=80)
                                else:
                                    output = query_openai_model(api_key, prompt)
                                flag = False
                            except openai.error.OpenAIError as e:
                                print("Some error happened here.")
                                time.sleep(1)
                        print(output)
                        outputs[key] = output
                fout.write('%s\n' % outputs)
                count += 1
                continue
            
            outputs = {}
            for key in prompts:
                prompt = prompts[key]
                flag = True
                while flag:
                    try:
                        if key in ['zs_resp', 'fs_resp']:
                            output = query_openai_model(api_key, prompt,max_tokens=40)
                        elif key in ['zs', 'zs-pcot']:
                            output = query_openai_model(api_key, prompt,max_tokens=80)
                        else:
                            output = query_openai_model(api_key, prompt)
                        flag = False
                    except openai.error.OpenAIError as e:
                        print("Some error happened here.")
                        time.sleep(1)
                print(output)
                outputs[key] = output
            fout.write('%s\n' % outputs)


if __name__ == "__main__":
    infer('otters-source.txt', 'otters-chatgpt.txt')
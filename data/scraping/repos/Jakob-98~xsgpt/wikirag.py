import requests as r
from openai_functools import openai_function

@openai_function
def f(title: str):
    """finds a wikipedia article by title"""
    S=r.Session();R=S.get(url="https://en.wikipedia.org/w/api.php",
                          params={"action":"query","format":"json","titles":title,"prop":"extracts","explaintext":"1",}).json()
    p=list(R['query']['pages'].keys())[0]
    return R['query']['pages'][p]['extract'] if p!="-1" else "Page does not exist"

def main():
    import openai, os, pickle, json
    from pathlib import Path
    c, h = (lambda m: openai.ChatCompletion.create(model='gpt-4', messages=m, functions=[f.openai_metadata], function_call="auto")), (lambda r, c, n=None: {"role": r, "content": c, "name": n} if r == "function" else {"role": r, "content": c})
    m = pickle.load(open(Path(os.getenv('APPDATA', os.path.expanduser('~')),'ai_chat.pkl'), 'rb')) \
        if Path(os.getenv('APPDATA', os.path.expanduser('~')),'ai_chat.pkl').exists() else [h('system', 'You are an AI assitant')]

    while (i := input('You: ').strip()) != 'exit':
        response = c((m:=[*m,h("user",i)]))
        response_message = response["choices"][0]["message"]
        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            function_args = json.loads(response_message["function_call"]["arguments"])
            function_response = f(**function_args)
            m.append(h("function", function_response, function_name))
            m.append(response_message)
            response = c(m)
        print(f'AI: {(r:=response["choices"][0]["message"]["content"])}')
        m = [*m, h("assistant", r)][-10:] # Truncate to 10 last messages
        pickle.dump(m, open(Path(os.getenv('APPDATA', os.path.expanduser('~')),'ai_chat.pkl'), 'wb'))


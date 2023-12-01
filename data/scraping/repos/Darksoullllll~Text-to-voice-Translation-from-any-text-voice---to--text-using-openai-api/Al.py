import openai
from api_key import openai_key
openai.api_key = openai_key

def text_translation(a,froml,tol):
    prompt = a+"translate this from "+froml+"to"+tol

    result =  openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]

    )
    return result.choices[0]['message']['content']

if __name__=='__main__':
    a = "Hello this is india"
    translate = "German"
    text_translation(a,translate)

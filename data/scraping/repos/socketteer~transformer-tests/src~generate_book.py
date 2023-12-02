import openai

gr_seed = '''Gravity's Rainbow is about the design, production and dispatch of V-2 rockets by the German 
military. In particular, it features the quest undertaken by several characters to uncover the secret of a mysterious 
device named the "Schwarzgerät" ("black device"), slated to be installed in a rocket with the serial number "00000".'''

gatsby_seed = '''The Great Gatsby is a 1925 Jazz Age novel about the impossibility of recapturing the past.'''


# infinite library prompt
# dialogue prompt
# wikipedia

def synopsis_dialogue(seed, max_tokens=300, n=1, temperature=0.85, engine='ada'):
    prompt = f'''"What is the book about?"
"{seed}"
"What happens in it?"
"'''
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        echo=False,
        top_p=1,
        n=n,
        stop=['"\n'],
        timeout=15
    )
    if n == 1:
        return response.choices[0]['text']
    else:
        synopses = []
        for choice in response.choices:
            synopses.append(choice['text'])
        return synopses


print(synopsis_dialogue(seed=gatsby_seed, engine='davinci'))
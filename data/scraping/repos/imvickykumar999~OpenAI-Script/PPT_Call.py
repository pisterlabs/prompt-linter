
import openai, os
import PPT_Fun as ppt

title = input('Enter PPT Title : ')
save_path = f'static/{title}.pptx'

try:
    os.mkdir('static')
except:
    pass

try:
    os.mkdir('article')
except:
    pass

def create_ppt():
    with open(f'article/{title}.txt') as f:
        arti = f.read()

    for i in arti.split('\n\n'):
        j = i.split('\n')

        if j:
            topic = j[0]
            print(topic)


            for k in j[1:]:
                try:
                    bullet = k.split('- ')[1]
                except:
                    bullet = k
                print('-', bullet)

                ppt.bullet_level(
                    path_to_presentation = save_path,
                    title_shape_text = title.upper(),
                    tf_text = topic,
                    p_text_l1 = bullet,
                    )

def make_slides(
        topic = "save water",
        API_Key = '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',
    ):
    
    openai.api_key = API_Key
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": f'''Write PPT on {topic}'''}])

    article = completion.choices[0].message.content
    with open(f'article/{title}.txt', 'w') as f:
        f.write(article)
                    
    return article

print(make_slides(title))
create_ppt()

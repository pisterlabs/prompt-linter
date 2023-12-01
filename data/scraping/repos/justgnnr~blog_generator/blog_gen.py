import os
import openai
import requests 
import datetime


openai.api_key = os.getenv("OPENAI_API_KEY")

def get_poem():
    poem_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Write a haiku poem about nature and technology."}])
    poem = poem_completion.choices[0].message.content
    title_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Provide a title for this poem. {poem}"}])
    title = title_completion.choices[0].message.content
    return (title, poem)

def get_img(path):
    get_img_prompt = f"In 20 words or less, suggest a prompt for creating an inspiring nature image. Suggest visual artists by name and styles."
    img_prompt_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": get_img_prompt}])
    img = openai.Image.create(
      prompt= img_prompt_completion.choices[0].message.content,
      n=2,
      size="1024x1024"
    )
    url = img.data[0].url
    r = requests.get(url, allow_redirects=True)
    open(f'{path}/header.png', 'wb').write(r.content)

def create_content():
    path = "../skinnydip/content/blog/"
    date = datetime.datetime.now().strftime("%Y-%m-%d")    
    new_post_path = os.path.join(path, date)
    os.mkdir(new_post_path)
    get_img(new_post_path)
    title, poem = get_poem()
    create_post(title, poem, date, new_post_path)

def create_post(title, poem, date, path):
    f = open(f'{path}/index.md', "w")
    header = f'---\ntitle: {title}\ndate: \"{date}\"\nfeaturedImage: header.png \n---'
    f.write(header)
    f.write('\n')
    f.write('\n')
    f.write('![nature image](./header.png)')
    f.write('\n')
    f.write('\n')
    f.write(poem)
    f.close()

    

if __name__ == "__main__":
    create_content()
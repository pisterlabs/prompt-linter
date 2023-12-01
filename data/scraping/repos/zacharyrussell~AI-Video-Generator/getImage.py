
import openai

def getImage(textResponse, name):
    response = openai.Image.create(
    prompt=textResponse,
    n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']
    # print(image_url)
    import requests
    img_data = requests.get(image_url).content
    with open(f'./frames/{name}.jpg', 'wb') as handler:
        handler.write(img_data)
    return f'./frames/{name}.jpg'
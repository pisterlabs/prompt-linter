import openai
def main(DATA):
    secret = ''
    openai.api_key = ''
    response = openai.Image.create(
    prompt=DATA,
    n=1,
    size="1024x1024"
    )
    return response['data'][0]['url']



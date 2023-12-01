import openai

response = openai.Image.create(
    prompt='Duck with glasses',
    n=1,
    size="1024x1024"
)

print(response['data'][0]['url'])

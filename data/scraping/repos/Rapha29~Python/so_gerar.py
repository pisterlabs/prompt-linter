import openai as ia # pip install openai

ia.api_key = 'coloca aqui a api'  #USE A MESMA API DO CHAT GPT

request = input('Descreva a imagem a ser gerada: ')

response = ia.Image.create(
    prompt = request,
    n=1,
    size="1024x1024"
)
imagem_url = response['data'][0]['url']

print(f'URL da imagem gerada: \n{imagem_url}')


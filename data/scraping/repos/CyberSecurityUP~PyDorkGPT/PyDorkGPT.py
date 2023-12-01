import openai
import requests
from urllib.parse import quote

# Define a chave da API do ChatGPT
openai.api_key = 'sua_chave_da_API_do_ChatGPT'

def generate_dork(prompt):
    # Gera uma dork usando a API do ChatGPT
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()

def google_search(query):
    # Faz a pesquisa no Google
    search_url = 'https://www.google.com/search?q=' + quote(query)
    response = requests.get(search_url)
    if response.status_code == 200:
        print('Resultados da pesquisa para:', query)
        print(response.text)
    else:
        print('Erro ao fazer a pesquisa.')

# Obtém o prompt de pesquisa do usuário
search_prompt = input('Digite o prompt de pesquisa: ')

# Verifica o formato do prompt e gera a dork ou usa o operador de pesquisa avançada do Google
if ':dork' in search_prompt:
    dork = search_prompt.replace(':dork', '').strip()
else:
    dork_prompt = 'site:.com.br intitle:index of'
    dork = generate_dork(dork_prompt)

print('Dork gerada:', dork)

# Faz a pesquisa no Google
google_search(dork)

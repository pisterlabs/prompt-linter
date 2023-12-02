import openai
import openai_secret_manager

# Configura a chave secreta da API
openai.api_key = openai_secret_manager.get_secret('openai')['api_key']

# Chama a API do GPT-3
response = openai.Completion.create(
    engine='davinci',
    prompt='The quick brown fox',
    max_tokens=5,
)

# Imprime a resposta
print(response.choices[0].text)

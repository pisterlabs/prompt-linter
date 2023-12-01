# Importando biblioteca
import openai

#chave de autenticação
openai.api_key = "SUA CHAVE"

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Quem foi Carl Sagan?",
    max_tokens=1000    
    )
print(response.choices[0].text)

import csv
import openai
import re

# Substitua "sk-..." pela sua chave de API secreta
openai.api_key = ""


model = "text-davinci-003"

# Leia os arquivos CSV usando o módulo csv do python e armazene os dados em uma lista
data = []
for filename in ["Data/Data.csv", "Data/Data2.csv", "Data/Data3.csv"]:
    with open(filename, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row[0])

# Crie uma função que recebe uma pergunta do usuário e gera um texto usando o modelo da open ai
def generate_text(question):
    # Construa a entrada para a API da open ai com os dados dos CSVs e a pergunta do usuário
    input = "Dados:\n"
    for row in data:
        input += row + "\n"
    input += "\nPergunta: " + question + "\n\nResposta:"

    # Envie uma requisição para a API da open ai com os parâmetros desejados
    response = openai.Completion.create(
        engine=model,
        prompt=input,
        temperature=0.9,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )

    # Retorne a resposta gerada pelo modelo
    return response["choices"][0]["text"]


# Teste a função com uma pergunta de exemplo
while True:
    question = input("Digite uma pergunta: ")
    if question == "quit":
        break   
    answer = generate_text(question)
    print(answer)

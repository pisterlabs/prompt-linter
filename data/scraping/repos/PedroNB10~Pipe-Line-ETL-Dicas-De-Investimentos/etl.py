import openai
import csv
import time


# Extraction
def ler_arquivo(arquivo):
    arquivo_csv = open(arquivo, mode='r')
    leitor_csv = csv.DictReader(arquivo_csv)

# Lista para armazenar os dicionários dos usuários
    lista_de_usuarios = []

# Itere pelas linhas do arquivo CSV
    for linha in leitor_csv:
        usuario = {}
        usuario['nome'] = linha['nome']
        usuario['Perfil'] = linha['Perfil']
        usuario['decisao'] = linha['decisao']
        
    # Adicione o dicionário do usuário à lista
        lista_de_usuarios.append(usuario)
    
    return lista_de_usuarios
        
 
# Transform
def gerar_mensagem(user):

    openai.api_key = "*********************************************"
    # Requisição para a API do chatgpt
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[ # Contexto de como o chatgpt deve ser
        {
            "role": "system",
            "content": f"Você é um especialista em investimentos e fará uma mensagem direcionada a um cliente que será solicitada."
        },
        
        { # Contexto para gerar mensagem
            "role": "system",
            "content": f"Para fazer a mensagem saiba que você terá que classificar o usuário em uma das seguintes categorias de tipo de investidor: conservador, arrojado ou moderado. Você terá as seguintes informações: o perfil do usuário que pode ser : Iniciante no assunto de finanças, Tradcional ou Antenado. E também é necesário que saiba que em uma situação de queda de 10% em um dos investimento do usário, ele Manter o valor investido, Vender o valor investido ou Aumentar o valor investido, isso pode ajudar a você elaborar melhor a mensagem e classíficá-lo em uma das categorias de tipo de investidor."
        },
                
        { # Solicitar a mensagem
            "role": "user",
            "content": f"Classifique o usuário {user['nome']} em uma das categorias de tipo de investidor: conservador, arrojado ou moderado, sabendo que : Ele tem perfil {user['Perfil']} e em uma situação de queda de 10% em um dos investimentos ele {user['decisao']} o valor investido. Como resposta, não diga as informações recebidas: perfil e situação de queda, diga ao usuário somente em que categoria se enquadra depois dê dicas de investimento a renda fixa, renda variável e fundos de investimento. Não pule linhas e faça isso em no máximo 400 caracteres."
        }
        ]
    )
    respostaChatgpt = completion.choices[0].message.content.strip('\"')
    return respostaChatgpt  # Retornar a mensagem gerada em formato de string remove as aspas duplas

# Load
def escrever_no_arquivo(usuario,mensagem):
    arquivo_csv  = open("Banco_de_respostas.csv", mode = 'a')
    arquivo_csv.write(f"Resposta do usuário {usuario['nome']}:\n {mensagem}\n\n")
    return arquivo_csv


if __name__ == "__main__":

    users = ler_arquivo("SDW2023.csv")

    for  user in users:
        print()
        print(f"Nome do usuário: {user['nome']}\nPerfil: {user['Perfil']}\nDecisão: {user['decisao']} o valor investido em uma situação de queda de 10% em um dos investimentos.")
        print()
        mensagem = gerar_mensagem(user)
        print(mensagem)
        arquivo_leitura = escrever_no_arquivo(user,mensagem)
        time.sleep(21) # ("Esperando 21 segundos para não exceder o limite de requisições da API -> 3 / min")


    arquivo_leitura.close()



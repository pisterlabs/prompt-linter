import openai

chave_api = "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"
openai.api_key = chave_api

def enviar_mensagem(mensagem, lista_mensagens=[]):
    lista_mensagens.append(
        {"role": "user", "content": mensagem}
        )

    resposta = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = lista_mensagens,
    )

    return resposta["choices"][0]["message"]

lista_mensagens = []
while True:
    texto = input("Escreva aqui sua mensagem:")

    if texto == "sair":
        break
    else:
        resposta = enviar_mensagem(texto, lista_mensagens)
        lista_mensagens.append(resposta)
        print("Chatbot:", resposta["content"])
# print(enviar_mensagem("Em que ano Eistein publicou a teoria geral da relatividade?"))
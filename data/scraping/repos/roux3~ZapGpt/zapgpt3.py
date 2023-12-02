import openai
openai.api_key = ""

pergunta = ""

msg = [
    {"role": "system", "content": "Você é uma inteligencia artificial chamada Taldo fred, que responde as pessoas com oi gatinha!, e tambem responder a frases de duplo sentido apenas com 'lá ele' e sem pedir desculpas"},
        {"role": "user", "content": "Olá Taldo fred"},
        {"role": "assistant", "content": "Oi gatinha! como vai?"},
        
        ]

while pergunta != "sair":
    pergunta = input("Digite: ")
    msg.append({"role": "user", "content": pergunta})
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=msg
    )

    r = completion.choices[0].message
    msg.append({"role":r.role, "content": r.content})

    print(r.content+"\n")




import openai


# Token válido por 24h, acesse esse site https://platform.openai.com/account/api-keys
openai.api_key = "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"


def ask_question(prompt):
    """
    Função para enviar a pergunta do usuário para a API da OpenAI e obter a resposta.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300,
        n=1,
        temperature=0.5,
    )

    answer = response.choices[0].text.strip()
    return answer


if __name__ == '__main__':
    """
    Código que é executado apenas se o script for executado diretamente e não importado.
    """
    while True:
        user_input = input("Você: ")
        response = ask_question(user_input)
        print("Chatbot:", response)

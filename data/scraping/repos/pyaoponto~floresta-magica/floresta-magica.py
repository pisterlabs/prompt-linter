import os
import openai
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Pega o valor de "OPENAI_API_KEY"
api_key = os.getenv('OPENAI_API_KEY')

# Define a chave da API para o OpenAI
openai.api_key = api_key

def perguntar_gpt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()

def loop_jogo():
    cenario = "Você está na entrada da floresta. À sua frente, há duas trilhas: uma que parece levar mais fundo na escuridão da floresta e outra que segue em direção à chama distante."

    print("Você é um viajante corajoso que se encontra em uma floresta escura e misteriosa, onde a única fonte de luz é uma chama tremeluzente ao longe. A floresta é conhecida pelos aldeões locais como a 'Floresta Encantada' e é dito que abriga uma entidade mágica que concede desejos.")
    print("Para ter seu desejo concedido, você precisa passar por uma série de provas que testarão seu valor. As provas variam de resolver enigmas enigmáticos a enfrentar criaturas mágicas perigosas.")
    print("\n"+cenario)

    objetivo_do_jogo = "Objetivo: Navegar pela Floresta Encantada, superar as provas, encontrar a entidade mágica e ter seu desejo concedido."

    contextualizacao_inicial = "Jogo de rpg de texto. "+ objetivo_do_jogo + " " + cenario

    while True:
        # Pega a ação do jogador
        acao = input("\nO que você faz?\n> ")
        # Atualiza o diálogo com a ação do jogador
        dialogo = contextualizacao_inicial + "\nAção do Jogador: " + acao
        # Usa o GPT para gerar uma resposta à ação do jogador
        dialogo += "\nResposta da IA: "
        resposta = perguntar_gpt(dialogo)
        # Imprime a resposta
        print(f"\nResposta da IA: {resposta}")
        # Atualiza o contexto inicial com a ação e resposta mais recentes
        contextualizacao_inicial = "Jogo de rpg de texto. " + objetivo_do_jogo + "\nAção do Jogador: " + acao + "\nResposta da IA: " + resposta

if __name__ == "__main__":
    loop_jogo()

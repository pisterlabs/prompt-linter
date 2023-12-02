import os
import time
import openai
import dotenv
import tiktoken

def analise_sentimento(nome_do_produto):
    prompt_sistema = """
    Você é um analisador de sentimentos de avaliações de produtos.
    Escreva um parágrafo com até 50 palavras resumindo as avaliações e depois atribua qual o sentimento geral para o produto.
    Identifique também 3 pontos fortes e 3 pontos fracos identificados a partir das avaliações.

    #### Formato de saída
    Nome do produto:
    Resumo das avaliações:
    Sentimento geral: [aqui deve ser POSITIVO, NEUTRO ou NEGATIVO]
    Pontos fortes: [3 Bullet points]
    Pontos fracos: [3 Bullet points]
    """

    prompt_usuario = carrega(f"./dados/avaliacoes-{nome_do_produto}.txt")

    print(f"Iniciando a análise do produto: {nome_do_produto}")
    tentativas = 0
    tempo_de_espera = 2
    while tentativas < 3:
        try:

            tentativas += 1
            print(f"Tentativa numero {tentativas}")
        
            resposta = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": prompt_sistema
                    },
                    {
                        "role": "user",
                        "content": prompt_usuario
                    }
                ]
            )

            salva(f"./dados/analise-{nome_do_produto}", resposta.choices[0].message.content)
            print("Análise concluída com sucesso!")
            return 

        except openai.error.AuthenticationError as e:
            print(f"Erro de autenticacao: {e}")
        except openai.error.APIError as e:
            print(f"Erro de api: {e}")
            time.sleep(2)      
        except openai.error.RateLimitError as e:
            print(f"Erro de limite de taxa: {e}")
            time.sleep(tempo_de_espera)
            tempo_de_espera *= 2
        

def carrega(nome_do_arquivo):
    try:
        with open(nome_do_arquivo, "r") as arquivo:
            dados = arquivo.read()
            return dados
    except IOError as e:
        print(f"Erro: {e}")

def salva(nome_do_arquivo, conteudo):
    try:
        with open(nome_do_arquivo, "w", encoding="utf-8") as arquivo:
            arquivo.write(conteudo)
    except IOError as e:
            print(f"Erro ao salvar arquivo: {e}")


dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.system('clear')

lista_de_produtos = ["DVD player automotivo", "Esteira elétrica para fitness", "Grill elétrico para churrasco", "Mixer de sucos e vitaminas", "Tapete de yoga", "Miniatura de carro colecionável", "Balança de cozinha digital", "Jogo de copos e taças de cristal", "Tabuleiro de xadrez de madeira", "Boia inflável para piscina"]

for nome_do_produto in lista_de_produtos:
    analise_sentimento (nome_do_produto)

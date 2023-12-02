import openai
import os
import dotenv
import tiktoken
import time
def ler_arquivo(nome_arquivo):
    try:
        with open(nome_arquivo, 'r') as arquivo:
            dados = arquivo.read()
            return dados
    except:
        print('erro')

def adiciona_arquivo(nome_arquivo,dados):
    try:
        with open(nome_arquivo, 'w', encoding='utf-8') as arquivo:
            arquivo.write(dados)
    except:
        print('erro')

dotenv.load_dotenv()
openai.api_key = os.getenv('API_GPT')

def analisar_produtos(nome_produto):
    prompt_sistema = """
        Você é um analisador de avaliações de produto, e vai me gerar uma analisa resumida que vai falar uma breve 
        descrição do produto e outra dividida em topicos
        
        ### Formato de saída analise resumida:
        deve ser um paragrafo em até 50 palavras
        ### Formato de saída analise em topicos:
        Avaliação geral: [POSITVO/NEGATIVO/REGULAR]
        Pontos Positivos: [listar 3 em topicos]
        Pontos Negativos: [listar 3 em topicos]    
    """
    print(f'preparando analise {nome_produto}...')
    prompt_usuario = ler_arquivo(f"../dados/avaliacoes_{nome_produto}.txt")
    tentativas = 0
    tempo_exponencial = 5
    while tentativas < 3:
        tentativas += 1
        try:
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
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            adiciona_arquivo(f'../dados/analise_{nome_produto}', resposta.choices[0].message.content)
            print(f'analise {nome_produto} realizada com sucesso')
            return
        except openai.error.AuthenticationError as e:
            print(f'ERRO DE AUTENTICAÇÃO: {e}')
        except openai.error.APIError as e:
            print(f'ERRO DE API: {e}')
            time.sleep(5)
        except openai.error.RateLimitError as e:
            print(f'ERRO LIMITE DE TAXA: {e}')
            time.sleep(tempo_exponencial)
            tempo_exponencial *= 2


produtos = ['balança','boia_inflavel','DVD','esteira','grill','jogo_de_copos','miniaturas_carros','mixer','tabuleiro_xadrez','tapete_yoga']

for i in produtos:
    analisar_produtos(i)
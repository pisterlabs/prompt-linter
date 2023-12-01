import os
import openai
import dotenv
import tiktoken
import time

dotenv.load_dotenv()

openai.api_type = "azure"
openai.api_base = "https://ciasc-openai.openai.azure.com/" #max tokens = 4096
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

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

def cria_jogo(prompt_user):
    eng = "ia_ciasc"  #modelo gpt-3.5-turbo
    tam_esperado_saida = 4000
    tentativa = 0
    tempo_de_tentativa = 5

    while tentativa <= 3:
        tentativa += 1
        try: 
            resposta = openai.ChatCompletion.create(
            engine=eng,
            messages=[
                {
                "role": "system",
                "content": f"""
                Você ajuda a desenvolver códigos em C utilizando chamadas de sistema e threads. A aplicação serve para implementar threads, exclusão mútua e coordenação de processos por meio de um jogo simples, onde os componentes devem rodar ao mesmo tempo e interagir com as regras de jogo.
                Considere que será rodado em ambiente Windows.
                O formato de saída é apenas o código em C
                """
                },
                {
                    "role": "user",
                    "content": prompt_user
                }
            ],
            temperature=1.2,
            max_tokens=8000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            )
            salva("./codigo_gpt.c", resposta.choices[0].message.content)
            print("Relatório criado com sucesso!")
            return
        # Tratamento de ERROS:
        except openai.error.AuthenticationError as e:
            print("Erro de Autenticação:", e)
        except openai.error.APIError as e:
            print("Erro de API:", e)
            if tentativa != 3:
                print("Aguarde. Tentando requisição novamente...")
            time.sleep(15)
        except openai.error.RateLimitError as e:
            print("Erro de taxa limite de requisição:", e)
            tempo_de_tentativa *= 2 #tecnica usada para não exagerar nas requisições

def verifica_codigo(codigo):
    eng = "ia_ciasc"  #modelo gpt-3.5-turbo
    tam_esperado_saida = 4000
    tentativa = 0
    tempo_de_tentativa = 5

    while tentativa <= 3:
        tentativa += 1
        try: 
            resposta = openai.ChatCompletion.create(
            engine=eng,
            messages=[
                {
                "role": "system",
                "content": f"""
                Você ajuda a corrigir códigos em C. Verifique se o codigo {codigo} funciona perfeitamente.
                O formato de saída é apenas o codigo corrigido e otimizado.
                """
                }
            ],
            temperature=1.2,
            max_tokens=8000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            )
            salva("./codigo_gpt_corrigido.c", resposta.choices[0].message.content)
            print("Relatório criado com sucesso!")
            return
        # Tratamento de ERROS:
        except openai.error.AuthenticationError as e:
            print("Erro de Autenticação:", e)
        except openai.error.APIError as e:
            print("Erro de API:", e)
            if tentativa != 3:
                print("Aguarde. Tentando requisição novamente...")
            time.sleep(15)
        except openai.error.RateLimitError as e:
            print("Erro de taxa limite de requisição:", e)
            tempo_de_tentativa *= 2 #tecnica usada para não exagerar nas requisições  

interface_velha = carrega("./prof.c")
codigo_bruto = carrega("./codigo_gpt.c")
#codigo_bruto = cria_jogo("Crie um jogo que exista um helicoptero que leva pessoas de um lado para o outro utilizando as setas do teclado")
verifica_codigo(codigo_bruto)
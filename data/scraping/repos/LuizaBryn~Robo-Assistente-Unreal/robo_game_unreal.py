import os
import openai
import dotenv
import tiktoken
import time
import json

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

def gerador_voz(prompt_usuario):
    eng = "ia_ciasc"  #modelo gpt-3.5-turbo
    tam_esperado_saida = 4000
    tentativa = 0
    tempo_de_tentativa = 5
    
    codificador = tiktoken.encoding_for_model("gpt-3.5-turbo")
    lista_tokens = codificador.encode(prompt_usuario)
    nro_tokens = len(lista_tokens)
    print(f"Número de tokens de entrada:{nro_tokens}")

    if nro_tokens >= 8000 - tam_esperado_saida:
      eng = "ia_ciasc_16k" #modelo gpt-3.5-turbo-16k
    print(f"Implementação escolhida: {eng}")

    while tentativa <= 3:
        tentativa += 1
        try: #INTEGRAÇÃO
            resposta = openai.ChatCompletion.create(
            engine=eng,
            messages=[
                {
                "role": "system",
                "content":  """ Você é um assistente virtual de uma empresa pública de informática e automação chamada CIASC.
                Você ajuda clientes com os produtos do CIASC.
                """
                },
                {
                    "role": "user",
                    "content": prompt_usuario
                }
            ],
            temperature=1.2,
            max_tokens=8000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            )
            resultado = resposta.choices[0].message.content
            #transformar 
            print("Resultado criado com sucesso!")
            return resultado

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

# ===============================================================================
# ============================== AQUI VOCÊ PODE EDITAR ==========================
## Encontra o txt e gera a resposta

transcricao_usuario = carrega("audio.mp3")
resposta_robo = gerador_voz(transcricao_usuario)

## Manda a resposta para a aplicação
## envia resposta_robo
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

def identificaPerfil(prompt_usuario):
    
    # ==================== VERIFICA QUAL MODELO DEVE SER USADO ====================
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
    # ==================== FIM DA VERIFICAÇÃO ====================

    # ================ USA REQUISIÇÃO ========================
    while tentativa <= 3:
        tentativa += 1
        try: 
            resposta = openai.ChatCompletion.create(
            engine=eng,
            messages=[
                {
                "role": "system",
                "content":  """
    Você é um analisador de gostos em animes. 
    Relacione: animes preferidos("1° Lugar" ao "10° Lugar"), 
    animes odiados("Anime ruim" e "Anime péssimo") e 
    "Preferência" da pessoa.
    Resuma o gosto da pessoa entre 5 e 10 palavras   

    ####Formato de saída deve ser em JSON:
    {
        "pessoa": [
        {
            "nickname": "nickname da pessoa",
            "animes_assistidos": "todos os animes listados pela pessoa",
            "animes_preferidos": "todos os animes citados em 1° Lugar, 2° Lugar, 3° Lugar, 4° Lugar, 5° Lugar, 6° Lugar, 7° Lugar, 8° Lugar, 9° Lugar e/ou 10° Lugar",
            "animes_odiados": "todos os animes citados em 'anime ruim' e 'anime péssimo'",
            "gosto": "o gosto da pessoa de acordo com as preferencias e os animes preferidos e os animes odiados em 5 palavras"
        }
        ]
    }

    ###EXEMPLO
    {
        "pessoa": [
        {
            "nickname": JoaozinhoSilva,
            "animes_assistidos": Another, One Piece, Hunter x Hunter, Shiki, Naruto, Haikyuu, Attack on Tintan.,
            "animes_preferidos": One Piece, Hunter x Hunter, Shiki, Haikyuu, Attack on Titan.,
            "animes_odiados": Another, Naruto.,
            "gosto": aventura, lições, lutas, reviravoltas, romance, shounen, seinen, amizade, imprevisivel, elaborado.
        }
        ]
    }
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
            json_resultado = json.loads(resultado)
            print("Resultado JSON criado com sucesso!")
            return json_resultado

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

def recomendaAnime(gosto, animes_assistidos, animes_p, animes_n, animes):
    
    # ==================== VERIFICA QUAL MODELO DEVE SER USADO ====================
    eng = "ia_ciasc"  #modelo gpt-3.5-turbo
    tentativa = 0
    tempo_de_tentativa = 5

    # ==================== FIM DA VERIFICAÇÃO ====================

    # ================ USA REQUISIÇÃO ========================
    while tentativa <= 3:
        tentativa += 1
        try: 
            resposta = openai.ChatCompletion.create(
            engine=eng,
            messages=[
                {
                "role": "system",
                "content": f"""
    Você é um recomendador de animes.
    Considere o seguinte gosto: {gosto}
    Considere os animes assistidos: {animes_assistidos}
    Considere os animes preferidos:{animes_p}
    Considere os animes odiados:{animes_n}
    Recomende 3 animes a partir da lista de animes válidos e que sejam adequados ao perfil informados e as regras informadas.

    ####Regras importantes:
    - JAMAIS recomende animes da lista de animes da pessoa
    - Não recomende animes parecidos com os animes odiados
    - Se possível, recomende animes parecidos com os animes preferidos

    ####Lista de animes válidos para recomendação (exceto os quais a pessoa já assistiu)
    {animes} 

    A saída deve ser apenas o nome dos animes recomendados em bullet points
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
            resultado = resposta.choices[0].message.content
            print("===========Término de recomendações==========")
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
      
def escreve_email(recomendacoes, nome):
    
    # ==================== VERIFICA QUAL MODELO DEVE SER USADO ====================
    eng = "ia_ciasc"  #modelo gpt-3.5-turbo
    tentativa = 0
    tempo_de_tentativa = 5

    # ==================== FIM DA VERIFICAÇÃO ====================

    # ================ USA REQUISIÇÃO ========================
    while tentativa <= 3:
        tentativa += 1
        try: 
            resposta = openai.ChatCompletion.create(
            engine=eng,
            messages=[
                {
                "role": "system",
                "content": f"""
                Escreva um e-mail recomendando os seguintes animes para {nome}:
                
                {recomendacoes}

                O e-mail deve ter no máximo 3 parágrafos.
                Se apresente como uma IA chamada AnIA que ama animes e esta em processo de desenvolvimento.
                Agradeça pela disposição da pessoa em participar da pesquisa.
                Peça de maneira amigável e educada para a pessoa preencher o formulário de Feedback para ajudar no melhoramento
                Link do formulário: https://forms.gle/eh82ykaPQjvU16Jf8
                O tom deve ser amigável, informal e descontraído.
                Trate o cliente como alguém próximo e conhecido
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
            resultado = resposta.choices[0].message.content
            print("===========Término de recomendações==========")
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
      
# ======================= DEFINICAO DE VARIAVEIS =======================
lista_animes = carrega("./dados/animes.csv")
# =========== COMEÇO DO PROGRAMA ============

print("Qual arquivo gostaria? \n 0 - Todos \n 1 a 11 - Grupo [sua escolha] ")
escolha_arq = int(input("Resposta: "))

if escolha_arq > 0 and escolha_arq <= 11:
    arquivo = f"./dados/dados_grupo{escolha_arq}.csv"
elif escolha_arq == 0:
    arquivo = f"./dados/dados_otakus.csv"
else:
    print("Escolha corretamente")

prompt_user = carrega(arquivo)
perfis = identificaPerfil(prompt_user)
for perfil in perfis["pessoa"]:
    p_nickname = perfil["nickname"]
    print(f"Recomendação para: {p_nickname}")
    recomendacoes = recomendaAnime(perfil["gosto"],
                                       perfil["animes_assistidos"],
                                       perfil["animes_preferidos"],
                                       perfil["animes_odiados"],
                                       lista_animes)
    email = escreve_email(recomendacoes, p_nickname)
    salva(f"./emails/primeiro-email-{p_nickname}.txt", email)


# importação de biblioteca externa
import os
import openai
import dotenv
import tiktoken

# Definiçãod e função que carrega o arquivo na variável dados, com lançamento de exceção
def carrega(nome_do_arquivo):
    # Código a ser executado sobre a abertura e atribuição do arquivo à uma variável
    try:
        with open(nome_do_arquivo, "r") as arquivo:
            dados = arquivo.read()
            return dados
    # Lançamento de exceção
    except IOError as e:
        print(f"Erro: {e}")

# Chamada da funçã que carrega as variáveis de ambiente
dotenv.load_dotenv()
# Atribuição do valor da variável de ambiente OPENAI_API_KEY ao atributo api_key do objeto openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# Atribuiçao de long string à variável de prompt do sistema
prompt_sistema = """
Identifique o perfil de compra para cada cliente a seguir.

O formato de saída deve ser:

cliente - descreva o perfil do cliente em 3 palavras
"""

# Atribuiçao do retorno da função carrega à variável de prompt do usuário
prompt_usuario = carrega(".venv\gpt-python-1-dados\lista_de_compras_100_clientes.csv")

# Processo de definição do encoder para o modelo desejado, atribuição da lista de tokens 
# retornada pela função .encode(str) e exibição da quantidade de tokens definidos
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
lista_tokens = encoder.encode(prompt_sistema + prompt_usuario)
print(f"\nQuantidade de tokens na entrada: {len(lista_tokens)}")

# Declaração de variável para o modelo, condicional que verificará qual tipo de modelo é mais adequado, 
# atribuindo o valor de cada condição à variável modelo e exibição do modelo escolhido
modelo = ""
tokens_saida = 2048
if len(lista_tokens) >= 4096 - tokens_saida:
    modelo = "gpt-3.5-turbo-16k"
else:
    modelo = "gpt-3.5-turbo"
print(f"Modelo escolhido: {modelo}\n")

# Construção do chat para envio dos prmpts e atribuição do retorno à variável
response = openai.ChatCompletion.create(
    # Atribuição do modelo de GPT escolhido
    model=modelo,
    # Atribuição da smensagens enviadas com os papéis definidos
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
    # Atribuição de valores aos atributos de configuração de resposta do GPT
    temperature=1,
    max_tokens=tokens_saida,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# Exibição da resposta
print(response.choices[0].message.content)
# Exibição do totald e tokens usados
print(f"Tokens usados:{response.usage.total_tokens}")

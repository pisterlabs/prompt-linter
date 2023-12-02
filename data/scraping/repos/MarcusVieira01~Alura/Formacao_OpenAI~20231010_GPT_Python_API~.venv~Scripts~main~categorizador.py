# Importação de bibliotecas externas
import openai
import dotenv
import os
import tiktoken

# Definição de função que categoriza produtos utilizando o modelo GPT 3.5 Turbo da OpenAI
def categorizaProduto(nome_produto, categorias_validas):
    # Atribuição de uma longstring à variável sendo o prompt de configuração do sistema
    prompt_sistema = f""" 
                        Você é um caracterizador de produtos. Usando somente a listagem abaixo 
                        de categorias válidas, categorize o produto inserido 
                        ###Lista de categorias###
                        {categorias_validas}
                        ###Exemplo###
                        Bola de tênis
                        Esportes
                      """

    # Criação, usando o objeto ChatCompletion, de um chat com a IA da OpenAI
    response = openai.ChatCompletion.create(
        # Definição do modelo de IA a ser usado
        model = "gpt-3.5-turbo-16k",
        # Definição de um array de dicionários com os parâmetros de configuração
        messages = [
            {
                "role":"system",
                "content":prompt_sistema
            },
            # Definição de dicionário para o prompt do usuário
            {
                "role":"user",
                "content":nome_produto
            }
        ],
        # Declaração dos parâmetros de configuração da resposta
        temperature = 1,
        max_tokens = 256,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0
    )

    print(response.choices[0].message.content)

    print(f"Tokens usados:{response.usage.total_tokens}")


# Atribui à variável o retorno da função que define o codificador para leitura da quantidade de 
codificador = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
# Atribuição do retorno da função .encode, sendo uma lista
lista_tokens = codificador.encode("""
                                     Você é um caracterizador de produtos. Usando somente a listagem abaixo 
                                     de categorias válidas, categorize o produto inserido.
                                  """)

# Chamada da função load_dotenv() para carga das variáveis de ambiente
dotenv.load_dotenv()

# Chamada de atributo e atribuição do valor da API key da OpenAI retornada pelo método getenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# Chamada de atributo e atribuição do valor da ORGANIZATION cadastrada dentro do OpenAI
openai.organization = os.getenv("ORGANIZATION")

# Exibe a quantidade e a lista de tokens
print(len(lista_tokens))
print(lista_tokens)

# Imprime mensagem e captura input do usuário, atribuindo o valor à variável com as categorias válidas
print("Digite as categorias válidas:")
categorias_validas = input()

# Loop while com condição de parada True, gerando um loop infinito com a evocação da função de categorização de produtos
while(True):
    # Imprime mensagem e captura input do usuário, atribuindo o valor à variável com o nome do produto à ser categorizado
    print("Digite o nome do produto:")
    nome_produto = input()

    # Evocação de função que categorizará os produtos inputados pelo usuário
    categorizaProduto(nome_produto, categorias_validas)



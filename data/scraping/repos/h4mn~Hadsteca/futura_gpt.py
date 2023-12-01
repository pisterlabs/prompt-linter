import os
import openai
import tiktoken
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def calculate_tokens(string, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def do_completion(prompt):
    reponse = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return reponse['choices'][0]['text']

def do_context(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        page_titulo = soup.title.string
        content_titulo = soup.find("h1", class_="firstHeading").string
        content = soup.find("div", {"id": "mw-content-text"})
        for unwanted in content(["script", "style"]):
            unwanted.decompose()
        content = content.get_text()
    return f"""Contexto:
    ---
    Título da página: {page_titulo}
    ---
    Título do conteúdo: {content_titulo}
    ---
    Conteúdo: 
Cadastro Novos Clientes
Sistema: Futura Server
Versão: 2016.08.29
Caminho:
Cadastro > Cadastro > Clientes - Novos

Por que é especifico?
Desmembrada a tela de cadastro de clientes com esta nova tela, onde poderá apenas fazer os cadastros.

Regra de Negócio:
Criada nova tela "Clientes - Novos", onde podera fazer apenas a inclusao de novos clientes. Nesta tela nao pode permitir editar ou excluir os cadastros. Ao cadastrar um novo cliente, deve manter no campo "Cadastrado Por" o usuario que fez o cadastro e a data/hora. Os novos cadastros feitos nesta tela devem ficar em laranja na consulta, até que seja conferido na tela "Clientes - Manutencao e Revisao". O filtro TipoInformacao deve filtrar os novos clientes que ainda nao foram conferidos, e os clientes conferidos. Criada nova permissao para acessar esta tela: "CLIENTES - NOVOS"

Impactos:
Na tela Cadastro > Cadastro > Clientes - Dados Tecnicos, nao pode revisar os dados tecnicos ate que o cadastro seja conferido na tela "Clientes - Manutencao e Revisao"

Especificações referentes:
Nenhum.

Unit(s) Específica(s):
ifCad_NovoCliente

Detalhes Técnicos:
Nenhum.
    """


def main():
    pergunta = "Como cadastro um cliente novo neste sistema?"
    encoding_name = "cl100k_base"
    num_tokens = calculate_tokens(pergunta, encoding_name)
    print(num_tokens)
    prompt = f'''
    {do_context("http://wiki.futurasistemas.com.br/index.php?title=00001_-_FuturaServer")}
    ---
    Pergunta: {pergunta}'''
    print(prompt)
    resposta = do_completion(prompt)
    print(resposta)


if __name__ == "__main__":
    main()

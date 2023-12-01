# Nativos do Python
import os
import sys

# Bibliotecas Externas
from io import BytesIO
import concurrent.futures
import configparser
from PIL import Image, ImageTk
import PySimpleGUI as sg
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv

# Biblioteca Específica da API do ChatGPT
import openai


# Verifica se o script está sendo executado como um arquivo EXE compilado
if getattr(sys, 'frozen', False):
    # Carrega as chaves do arquivo config.env
    config = configparser.ConfigParser()
    config.read('config.env')
    bing_api_key = config.get('Keys', 'BING_API_KEY')
    openai.api_key = config.get('Keys', 'OPENAI_API_KEY')
else:
    # Carrega as chaves do arquivo .env se estiver executando diretamente no compilador
    load_dotenv('config.env')
    bing_api_key = os.getenv('BING_API_KEY')
    openai.api_key = os.getenv('OPENAI_API_KEY')

# Verifica se as chaves de API estão configuradas
if not bing_api_key and not openai.api_key:
    sg.popup('Erro: As chaves do Bing API e do ChatGPT não estão especificadas.\nConfigure suas chaves no arquivo config.env no mesmo diretório do programa.', title="Faltam as chaves... ;)")
    sys.exit(1)
elif not bing_api_key:
    sg.popup('Erro: A chave do Bing API não está especificada.', title="Bing API")
    sys.exit(1)
elif not openai.api_key:
    sg.popup('Erro: A chave do ChatGPT não está especificada.', title="ChatGPT API")
    sys.exit(1)


def construir_prompt(nome, idade, moradia, humor, personagem, pergunta):
    faixas_etarias = {
        (0, 3): "bebê",
        (4, 10): "criança",
        (11, 17): "adolescente",
        (18, 44): "adulto",
        (45, 59): "meia idade",
        (60, 74): "idoso",
        (75, float("inf")): "ancião"
    }

    faixa_etaria = next(descricao for faixa, descricao in faixas_etarias.items() if faixa[0] <= idade <= faixa[1])

    prompt = f"ChatGPT, imagine que você é {personagem}. Você deve responder e conversar como esse personagem faria, utilizando gírias e palavras típicas conhecidas desse personagem. Considere que quem lhe emite a mensagem ou pergunta é alguém chamado(a) {nome}, que tem {idade} anos de idade (se a for {faixa_etaria}, trate-o(a) como tal, ou seja, respondendo de forma adequada para sua faixa etária), mora em {moradia} e está se sentindo {humor}. Essa pessoa lhe diz: '{pergunta}'. Qual seria a sua resposta, ou seja, como o(a) {personagem} responderia em primeira pessoa?"

    return prompt


# Função para enviar a solicitação à API do OpenAI
def enviar_solicitacao(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.2,
            max_tokens=300,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=1,
        )
        answer = response.choices[0].text.strip()
        return answer
    except openai.OpenAIError as e:
        # Trate erros específicos da API do OpenAI
        raise Exception(f"Erro ao conversar com o personagem: {str(e)}")


# Função principal para conversar com o personagem
def conversar_com_personagem(nome, idade, moradia, humor, personagem, pergunta):
    prompt = construir_prompt(nome, idade, moradia, humor, personagem, pergunta)
    resposta = enviar_solicitacao(prompt)
    return resposta


# Em caso de exceção na função, tenta mais vezes
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))

# Função para obter a imagem do personagem
def download_personagem_image(query):
    # Usa chavea
    subscription_key = bing_api_key
    # Parametriza a consulta na API
    search_url = "https://api.bing.microsoft.com/v7.0/images/search"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
        # imagem do personagem, busca 3 imagens pelos menos (se for só uma pode dar erro), aplicava filtro adequado conforme idade
    params = {"q": query, "count": 3, "safesearch": filtro_familia}
    
    # Obtém resposta
    response = requests.get(search_url, headers=headers, params=params, timeout=5)
    response.raise_for_status()
    search_results = response.json()

    # Trata a resposta
    for result in search_results["value"]:
        image_url = result["thumbnailUrl"]
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image
        # Trata erros na requisição http
        except requests.exceptions.RequestException:
            continue
    # Trata outros erros
    raise Exception("Não foi possível encontrar uma imagem válida para o personagem.")


# Monta layout da tela do programa com o PySimpleGUI
layout = [
    [sg.Text("Qual o seu nome? "), sg.Input(key='-NOME-', size=(30,), enable_events=False)],
    [sg.Text("Qual a sua idade? "), sg.Input(key='-IDADE-', size=(30,), enable_events=False)],
    [sg.Text("Onde você mora? "), sg.Input(key='-MORADIA-', size=(30,), enable_events=False)],
    [sg.Text("Como você está se sentindo agora? "), sg.Input(key='-HUMOR-', size=(30,), enable_events=False)],
    [sg.Text("Com quem você quer falar? "), sg.Input(key='-PERSONAGEM-', size=(30,), enable_events=False)],
    [sg.Text("Escreva a mensagem, diga ou pergunte algo: "), sg.Input(key='-PERGUNTA-', size=(75,), enable_events=False)],
    [sg.Button("Enviar mensagem ou pergunta"), sg.Button("Sair")],
    [sg.Image(key='-IMAGE-', size=(201, 201)), sg.Output(size=(81, 21), key='-OUTPUT-', expand_x=True, expand_y=True)],
]


# Título da janela do programa
window = sg.Window("Simulador de Personagens - by René - Versão 2.0", layout, resizable=True)

# Ativa executor para rodar as duas consultar simultaneamente, uma thread cada
executor = concurrent.futures.ThreadPoolExecutor()


# Funcional
while True:
    try:
        event, values = window.read(timeout=100)

        if event == sg.WIN_CLOSED or event == "Sair":
            break
        elif event == "Enviar mensagem ou pergunta":
            nome = values['-NOME-']
            try:
                idade = int(values['-IDADE-'])
            except ValueError:
                sg.popup('Erro: Insira um valor numérico para a idade', title="Erro na idade")
                continue
            moradia = values['-MORADIA-']
            humor = values['-HUMOR-']
            personagem = values['-PERSONAGEM-']
            pergunta = values['-PERGUNTA-']

            resposta_future = executor.submit(conversar_com_personagem, nome, idade, moradia, humor, personagem, pergunta)
            
            idade = int(idade)
            if idade <= 17:
                filtro_familia = "Strict"
            elif 18 <= idade >= 59:
                filtro_familia = "Off"
            else:
                filtro_familia = "Moderate"
            image_future = executor.submit(download_personagem_image, personagem)

            resposta = resposta_future.result()
            window['-OUTPUT-'].print(f"{personagem.upper()} DIZ:\n{resposta}\n\n")
            
            image = image_future.result()
            if image is not None:
                image.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(image)
                window['-IMAGE-'].update(data=photo)
            else:
                window['-IMAGE-'].update(data=None)

    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

# Encerra o programa
window.close()
executor.shutdown()

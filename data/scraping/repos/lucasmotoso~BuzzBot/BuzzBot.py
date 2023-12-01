from selenium import webdriver
import time
import os
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import requests
import openai
import PySimpleGUI as sg 

required_libraries = ["selenium", "PySimpleGUI","webdriver"],
#Tema
sg.theme('GreenMono')



# Configurações iniciais
agent = {"User-Agent": 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}

# Obter informações da API Editacodigo
api = requests.get("https://editacodigo.com.br/index/api-whatsapp/sQenDpXGbdFd6PHJajx6Fsd4464IQzc8", headers=agent)
time.sleep(1)
api = api.text
api = api.split(".n.")
token1 = api[0].strip()
token2 = api[1].strip()
token3 = api[2].strip()
bolinha_notificacao = api[3].strip()
contato_cliente = api[4].strip()
caixa_msg = api[5].strip()
msg_cliente = api[6].strip()

# Função do Bot
def bot():
    try:
        # ... código para pegar e processar as mensagens ...

        # Clicar na notificação
        bolinha = driver.find_element(By.CLASS_NAME, bolinha_notificacao)
        bolinha = driver.find_elements(By.CLASS_NAME, bolinha_notificacao)
        clica_bolinha = bolinha[-1]
        acao_bolinha = webdriver.common.action_chains.ActionChains(driver)
        acao_bolinha.move_to_element_with_offset(clica_bolinha, 0, -20)
        acao_bolinha.click()
        acao_bolinha.perform()
        acao_bolinha.click()
        acao_bolinha.perform()

        # Ler a nova mensagem
        todas_as_msg = driver.find_elements(By.CLASS_NAME, msg_cliente)
        todas_as_msg_texto = [e.text for e in todas_as_msg]
        msg = todas_as_msg_texto[-1]
        print(msg)

        cliente = 'Mensagem do Cliente'
        texto2 = 'Responda a mensagem do cliente com base no próximo texto'
        questao = cliente + msg + texto2 + texto

        # Processar mensagem na API IA
        openai.api_key = apiopenai.strip()

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=questao,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        resposta = response['choices'][0]['text']
        print(resposta)
        time.sleep(3)

        # Responder à mensagem
        campo_de_texto = driver.find_element(By.XPATH, caixa_msg)
        campo_de_texto.click()
        time.sleep(3)
        campo_de_texto.send_keys(resposta, Keys.ENTER)
        time.sleep(10)

        # Fechar o contato
        webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()

    except openai.error.OpenAIError as e:
        print("Erro na API do OpenAI", e)
    except Exception as e:
        print('Erro', str(e))
        print('Buscando novas notificações')

# Interface Gráfica

imagem = sg.Image(filename='BuzzBot.png', key='-IMAGE-',)
imagem2 = sg.Image(filename='BuzzBot.png', key='-IMAGE-',)
tela = sg.Image(filename='background.jpeg', key='-IMAGE-',)

layout_login = [
    
    [sg.Text('Bem Vindo(a) ao BuzzBot -  Assistente para o WhatsApp')],
    [sg.Column([[imagem]],justification='center')],
    [sg.Text('Senha')],
    [sg.Input(key='senha', password_char='*')],
    [sg.Button('Entrar')],
    [sg.Text('',key='mensagem')],
    [sg.Text('Desenvolvido por LoremWeb - loremweb.com.br - 2023')],
    

]

layout_main = [
    [sg.Text('Bem vindo ao BuzzBot By: LoremWeb')],
    [sg.Column([[imagem2]],justification='center',)],
    [sg.Text('Insira a sua API da OPENAI')],
    [sg.Input(key='apiopenai', default_text='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')],
    [sg.Text('Ensine com linguagem humana, as intruções que o BuzzBot deverá fazer no WhatsApp')],  # Carrega a API salva automaticamente
    [sg.Multiline(size=(80, 20), key='texto')],
    [sg.Text('Tenha o celular em mãos com um número de WhatsApp válido')],
    [sg.Button('Capturar QR Code')],
    [sg.Text('Desenvolvido por LoremWeb - loremweb.com.br - 2023')],
]

window_login = sg.Window('BuzzBot - Login', layout=layout_login)
window_main = sg.Window('BuzzBot - Principal', layout=layout_main)



while True:
    event_login, values_login = window_login.read()
    if event_login == sg.WIN_CLOSED:
        break
    if event_login == 'Entrar':
        senha = values_login['senha']
        if senha == token1:
            window_login.close()
            event_main, values_main = window_main.read()
            if event_main == 'Capturar QR Code':
                apiopenai = values_main['apiopenai']
                texto = values_main['texto']
                window_main.close()
                dir_path = os.getcwd()
                chrome_options2 = Options()
                chrome_options2.add_argument(r"user-data-dir=" + dir_path + "profile/zap")
                driver = webdriver.Chrome(options=chrome_options2)
                driver.get('https://web.whatsapp.com/')
                time.sleep(10)
                while True:
                    bot()

            if event_main  == sg.WIN_CLOSED:
                break
        else:
            window_login['mensagem'].update('Senha incorreta - Tente novamente ou contate o Desenvolvedor')

from selenium import webdriver
import time
import os
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import requests
import openai
import PySimpleGUI as sg



#######################
########  API
agent = {"User-Agent": 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}

api = requests.get("https://editacodigo.com.br/index/api-whatsapp/abdXz2NqlAlrEIEJutaSinvRr7p3uQ8Q" ,  headers=agent)
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

###############################################



##############################
############### BOT
def bot():

    try:
        ######PEGAR A MENSAGEM E CLICAR NELA
        bolinha = driver.find_element(By.CLASS_NAME,bolinha_notificacao)
        bolinha = driver.find_elements(By.CLASS_NAME,bolinha_notificacao)
        clica_bolinha = bolinha[-1]
        acao_bolinha =  webdriver.common.action_chains.ActionChains(driver)
        acao_bolinha.move_to_element_with_offset(clica_bolinha,0,-20)
        acao_bolinha.click()
        acao_bolinha.perform()
        acao_bolinha.click()
        acao_bolinha.perform()


        ##### LER A NOVA MSG _21Ahp
        todas_as_msg = driver.find_elements(By.CLASS_NAME,msg_cliente)
        todas_as_msg_texto = [e.text for e  in todas_as_msg]
        msg = todas_as_msg_texto[-1]
        print(msg)


        ####PROCESSA A MENSAGEM NA API ia
        sistema = 'explique tudo sobre o hotel Copacabana Palace. Endereço: Av. Atlântica, 1111 - Copacabana, Rio de Janeiro - RJ, 22021-111. Telefone: (21) 2548-1111, reservas por email: reserva@email.com, aceitamos todas as formas de pagamento. OBS: responda com no maximo com 15 palavras'
        #### PROCESSA A MENSAGEM NA API IA

        chave_api = apiopenai.strip()

        editacodigo = 'abdXz2NqlAlrEIEJutaSinvRr7p3uQ8Q'

        resposta = requests.get("https://editacodigo.com.br/gpt/index.php", params={'pagina': editacodigo,'sistema': sistema, 'chave_api': chave_api, 'mensagem_usuario': msg}, headers=agent)
        time.sleep(3)
        resposta = resposta.text  


        #RESPONDE A MSG
        campo_de_texto = driver.find_element(By.XPATH,caixa_msg)
        campo_de_texto.click()
        time.sleep(3)
        campo_de_texto.send_keys(resposta,Keys.ENTER)
        time.sleep(2)    




        #FECHA O CONTATO
        webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
        



     


        


    except:
        print('buscando novas notificações')




######################################################################################



##########################
######### TELA 

imagem = sg.Image(filename='ia.png', key='-IMAGE-')
imagem2 = sg.Image(filename='ia.png', key='_CAMIMAGE_')

tela1 = [
    [sg.Column([[imagem]],justification='center')],
    [sg.Text('SENHA')],
    [sg.Input(key='senha')],
    [sg.Button('ENTRAR')],    
    [sg.Text('',key='mensagem')],

]


tela2 = [
    [sg.Column([[imagem2]],justification='center')],
    [sg.Text('BEM VINDO AO BOT DE INTELGENCIA ARTIFICIAL')],
    [sg.Text('Insira a API da OPENAI')],
    [sg.Input(key='apiopenai')],
    [sg.Text('Insira o texto')],
    [sg.Multiline(size=(80,20),key='texto')],
    [sg.Text('TENHA O CELULAR EM MÃOS')],
    [sg.Text('CLIQUE ABAIXO PARA CAPTURAR O QR CODE')],
    [sg.Button('CAPTURAR QRCODE')],
  
]
#############################################################


windows1 = sg.Window('IA BOT', layout= tela1)
windows2 = sg.Window('IA BOT', layout= tela2)


while True:
    event, values = windows1.read()
    if event == sg.WIN_CLOSED:
        break
    if event == 'ENTRAR':
        senha = values['senha']
        if senha == token1 :
            windows1.close()
            event2, values2 = windows2.read()
            if event2 == 'CAPTURAR QRCODE':
                apiopenai = values2['apiopenai']
                texto = values2['texto']
                windows2.close()
                dir_path = os.getcwd()
                chrome_options2 = Options()
                chrome_options2.add_argument(r"user-data-dir=" + dir_path + "profile/zap")
                driver = webdriver.Chrome(options = chrome_options2)
                driver.get('https://web.whatsapp.com/')
                time.sleep(10)
                while True:
                    bot()

            if event2 == sg.WIN_CLOSED:
                break
        else:
            windows1['mensagem'].update('ERRO DE SENHA')

            


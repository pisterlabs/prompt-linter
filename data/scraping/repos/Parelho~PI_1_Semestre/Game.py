import pygame
import pygame_textinput
import time
import psycopg
import os
import random
import openai
import re
from dotenv import load_dotenv

# Inserir no arquivo .env a chave de api da openai para o nivel infinito funcionar
load_dotenv()
openai.api_key = os.getenv("api_key")

# Inicializa o pygame
pygame.init()
# Deixa o nome da janela como CodeQuiz
pygame.display.set_caption('CodeQuiz')
# Utilizados como workaround de um bug que estava impedindo a classe Login de pegar os valores atualizados de acertos, level e streak, se conseguir resolver o bug irei remover essa mostruosidade
acertos = 0
level = 0
streak = 0
coins = 0
logoff = False
boost = False
boost_ok = False
shield = False
shield_ok = False
fechar = False
cosmetico1_desbloqueado = False
cosmetico2_desbloqueado = False
cosmetico3_desbloqueado = False
cosmetico1_ok = False
cosmetico2_ok = False
cosmetico3_ok = False
mascote = pygame.image.load(os.path.join("imgs", "Mascote.png"))

# Gera o input do chatgpt pra gerar as pergutas e respostas do nivel infinito
def gerar_texto_chatgpt():
    try:
        global completion
        completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": "Uma pergunta simples com respostas curtas sobre programar em Python. de 4 alternativas, sendo apenas 1 correta informe a resposta correta na primeira alternativa"}
                    ]
                    )
    except:
        print("erro de conexao a api da openai")

# Constantes
win = pygame.display.set_mode((900,600))
clock = pygame.time.Clock()

FONT = pygame.font.SysFont("timesnewroman", 50)
FONT_LOGIN = pygame.font.SysFont("timesnewroman", 30)
FONT_MOEDAS = pygame.font.SysFont("comicsans", 35)
FONT_MASCOTE = pygame.font.SysFont("comicsans", 20)
FONT_PERGUNTA = pygame.font.SysFont("arial", 20)
FONT_NIVEL = pygame.font.SysFont("arial", 100)

# Classes
class Jogador:
    def __init__(self):
        self.tema = "white"
        self.tema_rect = pygame.Rect(675, 100, 200, 100)
        self.tema_azul_rect = pygame.Rect(675, 250, 200, 100)
        self.engrenagem_rect = pygame.Rect(20, 500, 100, 100)
        self.loja_rect = pygame.Rect(120, 500, 100, 100)
        self.voltar_rect = pygame.Rect(400, 500, 100, 30)
        self.boost_rect = pygame.Rect(20, 420, 128, 128)
        self.shield_rect = pygame.Rect(20, 220, 128, 128)
        self.logout_rect = pygame.Rect(20, 450, 128, 128)
        self.cosmetico1_rect = pygame.Rect(750, 100, 64, 64)
        self.cosmetico2_rect = pygame.Rect(750, 200, 64, 64)
        self.cosmetico3_rect = pygame.Rect(750, 300, 64, 64)
        self.opcoes_aberto = False
        self.loja_aberta = False
        self.login = Login()

    def menu_principal(self):
        #Loja
        loja = pygame.image.load(os.path.join("imgs", "Loja.png"))
        win.blit(loja, (120, 500))
        mpos = pygame.mouse.get_pos()

        if self.loja_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.loja_aberta = True
    
        #Mascote
        global mascote
        win.blit(mascote, (0, 50))

        mensagem = FONT_MASCOTE.render("Bem Vindo ao CodeQuiz!", True, "black")
        win.blit(mensagem, (0, 0))

        #Opcoes
        engrenagem = pygame.image.load(os.path.join("imgs", "engrenagem.png"))
        win.blit(engrenagem, (20, 500))

        if self.engrenagem_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.opcoes_aberto = True
            time.sleep(0.2)

    def opcoes(self):
        mpos = pygame.mouse.get_pos()
        if self.tema_rect.collidepoint(mpos):
            if pygame.mouse.get_pressed()[0]:
                self.tema = "white"
        elif self.tema_azul_rect.collidepoint(mpos):
            if pygame.mouse.get_pressed()[0]:
                self.tema = "cornflowerblue"

        pygame.draw.rect(win, "black",[670, 95, 210, 110])
        pygame.draw.rect(win, "white",[675, 100, 200, 100])
        pygame.draw.rect(win, "black",[670, 245, 210, 110])
        pygame.draw.rect(win, "cornflowerblue",[675, 250, 200, 100])

        temas = FONT_LOGIN.render("Clique para mudar de tema", True, "black")
        win.blit(temas,(450, 0))

        pygame.draw.rect(win, "black",[398, 498, 79, 34], 0, 3)
        pygame.draw.rect(win, "burlywood2",[400, 500, 75, 30], 0, 3)
        # Define o valor true que mantem uma tela nova aberta como false para voltar para a anterior
        voltar = FONT_LOGIN.render("Voltar", True, "white")
        win.blit(voltar,(400, 500))
        if self.voltar_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.opcoes_aberto = False
        
        logout = pygame.image.load(os.path.join("imgs", "Exit.png"))
        win.blit(logout, (20, 450))
        if self.logout_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.opcoes_aberto = False
            global logoff
            logoff = True
            global mascote
            mascote = pygame.image.load(os.path.join("imgs", "Mascote.png"))
            global cosmetico1_ok
            global cosmetico2_ok
            global cosmetico3_ok
            cosmetico1_ok = False
            cosmetico2_ok = False
            cosmetico3_ok = False
            time.sleep(0.2)
        
    def loja(self):
        mpos = pygame.mouse.get_pos()
        texto = FONT_LOGIN.render("Clique em um item para comprar", True, "black")
        win.blit(texto,(300, 0))
        powerups = FONT_LOGIN.render("Powerup = 100 moedas", True, "black")
        win.blit(powerups,(0, 50))
        cosmeticos = FONT_LOGIN.render("Cosmeticos = 200 moedas", True, "black")
        win.blit(cosmeticos,(550, 50))

        pygame.draw.rect(win, "black",[398, 498, 79, 34], 0, 3)
        pygame.draw.rect(win, "burlywood2",[400, 500, 75, 30], 0, 3)
        voltar = FONT_LOGIN.render("Voltar", True, "white")
        win.blit(voltar,(400, 500))
        if self.voltar_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.loja_aberta = False
        
        bonus = pygame.image.load(os.path.join("imgs", "Boost.png"))
        win.blit(bonus, (20, 420))
        bonus_texto = FONT_LOGIN.render("Boost de Pontos", True, "black")
        win.blit(bonus_texto,(150, 420))
        global boost
        if self.boost_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0] and boost == False:
            boost = True
        
        protecao = pygame.image.load(os.path.join("imgs", "shield.png"))
        win.blit(protecao, (20, 220))
        protecao_texto = FONT_LOGIN.render("Proteção de Streak", True, "black")
        win.blit(protecao_texto, (150, 220))
        global shield
        if self.shield_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0] and shield == False:
            shield = True
        
        global mascote
        win.blit(mascote, (450, 200))
        cosmetico1 = pygame.image.load(os.path.join("imgs", "cosmetic1.png"))
        cosmetico2 = pygame.image.load(os.path.join("imgs", "cosmetic2.png"))
        cosmetico3 = pygame.image.load(os.path.join("imgs", "cosmetic3.png"))
        win.blit(cosmetico1, (750, 100))
        win.blit(cosmetico2, (750, 200))
        win.blit(cosmetico3, (750, 300))
        global cosmetico1_desbloqueado
        global cosmetico2_desbloqueado
        global cosmetico3_desbloqueado
        global cosmetico1_ok
        global cosmetico2_ok
        global cosmetico3_ok
        if self.cosmetico1_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0] and cosmetico1_desbloqueado == False and cosmetico1_ok == False:
            cosmetico1_desbloqueado = True
        elif self.cosmetico1_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0] and cosmetico1_ok == True:
            mascote = pygame.image.load(os.path.join("imgs", "Mascote1.png"))

        if self.cosmetico2_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0] and cosmetico2_desbloqueado == False and cosmetico2_ok == False:
            cosmetico2_desbloqueado = True
        elif self.cosmetico2_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0] and cosmetico2_ok == True:
            mascote = pygame.image.load(os.path.join("imgs", "Mascote2.png"))

        if self.cosmetico3_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0] and cosmetico3_desbloqueado == False and cosmetico3_ok == False:
            cosmetico3_desbloqueado = True
        elif self.cosmetico3_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0] and cosmetico3_ok == True:
            mascote = pygame.image.load(os.path.join("imgs", "Mascote3.png"))

class SeletorDeNivel():
    def __init__(self):
        self.voltar_rect_pergunta = pygame.Rect(400, 500, 100, 30)
        self.lv1 = pygame.Rect(270, 70, 160, 160)
        self.lv2 = pygame.Rect(470, 70, 160, 160)
        self.lv3 = pygame.Rect(270, 245, 160, 160)
        self.lv_endless = pygame.Rect(470, 245, 160, 160)
        self.lv1_aberto = False
        self.lv2_aberto = False
        self.lv3_aberto = False
        self.lv_endless_aberto = False
        self.lv_aberto = False
        self.lv2_desbloqueado = False
        self.lv3_desbloqueado = False
    
    def selecionar_nivel(self, xp):
        # Desbloquea os niveis caso o jogador possua xp o suficiente
        if xp >= 1000:
            self.lv2_desbloqueado = True
            if xp >= 2000:
                self.lv3_desbloqueado = True

        # Cadeado é colocado em cima da bolha de um nível caso ele não esteja desbloqueado
        cadeado = pygame.image.load(os.path.join("imgs", "Lock.png"))
        pygame.draw.rect(win, "dimgrey",[250, 0, 5 ,600])
        pygame.draw.rect(win, "dimgrey",[650, 0, 5 ,600])
        win.blit(FONT_LOGIN.render("Selecionar nivel", True, "black"), (350, 0))
        pygame.draw.circle(win, "burlywood2",[350, 150], 80)
        win.blit(FONT_NIVEL.render("1", True, "white"), (325, 90))
        pygame.draw.circle(win, "burlywood2",[550, 325], 80)
        openai = pygame.image.load(os.path.join("imgs", "OPENAI.png"))
        win.blit(openai, (480, 255))
        
        # Verifica se o nivel esta desbloqueado para mostrar o cadeado ou o nivel aberto
        if self.lv2_desbloqueado:
            pygame.draw.circle(win, "burlywood2",[550, 150], 80)
            win.blit(FONT_NIVEL.render("2", True, "white"), (525, 90))
        else:
            pygame.draw.circle(win, "azure4",[550, 150], 80)
            win.blit(cadeado, (525, 125))
        if self.lv3_desbloqueado:
            pygame.draw.circle(win, "burlywood2",[350, 325], 80)
            win.blit(FONT_NIVEL.render("3", True, "white"), (325, 265))
        else:
            pygame.draw.circle(win, "azure4",[350, 325], 80)
            win.blit(cadeado, (325, 300))
        mpos = pygame.mouse.get_pos()

        # Verifica qual nivel esta sendo aberto
        if self.lv1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.lv_aberto = True
            self.lv1_aberto = True
        elif self.lv2.collidepoint(mpos) and pygame.mouse.get_pressed()[0] and xp >= 1000:
            self.lv_aberto = True
            self.lv2_aberto = True
        elif self.lv3.collidepoint(mpos) and pygame.mouse.get_pressed()[0] and xp >= 2000:
            self.lv_aberto = True
            self.lv3_aberto = True
        elif self.lv_endless.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.lv_aberto = True
            self.lv_endless_aberto = True
        
        # Deixa os valores de nivel aberto false caso o nivel seja fechado para parar de mostrar a tela do nivel em cima da tela do menu principal
        if self.lv_aberto == False:
            self.lv1_aberto = False
            self.lv2_aberto = False
            self.lv3_aberto = False
            self.lv_endless_aberto = False

class Pergunta(SeletorDeNivel, Jogador):
    def __init__(self):
        self.voltar_ok = False
        # Define as listas com as perguntas
        self.perguntas_lv1 = ["7 // 2 vale quanto?", "print 'Hello, ', 'world', tera qual resultado no console?'", "10 % 2 vale quanto?", "Qual o simbolo utilizado para adicionar comentarios?", "100 / 0 vale quanto?"]
        # Escolhe uma pergunta aleatória da lista
        self.lv1_index = random.randint(0, len(self.perguntas_lv1) - 1)
        self.perguntas_lv2 = ["print('Hello' + 'world') terá qual resultado?", "idade = 7 + 5 = 4, idade terá qual valor?", "7.5 // 2 vale quanto", "Como posso criar uma função em Python?", "Como posso contar a frequência de elementos em uma lista em Python?"]
        self.lv2_index = random.randint(0, len(self.perguntas_lv2) - 1)
        self.perguntas_lv3 = ["Como posso verificar se uma lista está vazia em Python?", "Como posso converter uma string em maiúsculas em Python?", "Como posso criar um dicionário vazio em Python?", "Como posso criar uma classe em python?", "Como faço para instalar um pacote externo em Python usando o pip?"]
        self.lv3_index = random.randint(0, len(self.perguntas_lv3) - 1)
        self.resp1 = pygame.Rect(10, 170, 200, 100)
        self.resp2 = pygame.Rect(250, 170, 200, 100)
        self.resp3 = pygame.Rect(10, 300, 200, 100)
        self.resp4 = pygame.Rect(250, 300, 200, 100)
        self.nova_pergunta = pygame.Rect(325, 425, 250, 30)
        self.resposta = Resposta()
        self.respostas_ok = False
        self.pergunta_ok = False
        self.correta = 0
        self.acerto = False
        self.erro = False
        self.shuffle_ok = False
        self.resp_certa = ""
        self.respostas = []
    
    def nivel(self, lv1_aberto, lv2_aberto, lv3_aberto, lv_endless_aberto, voltar_rect_pergunta, lv_aberto):
        troca_ok = False
        global level
        global acertos
        global streak
        global shield
        win.blit(FONT_LOGIN.render("Streak: " + str(streak), True, "black"), (600, 0))
        mpos = pygame.mouse.get_pos()
        if lv1_aberto:
            level = 1
            pygame.draw.rect(win, "azure4",[10, 170, 200, 100])
            pygame.draw.rect(win, "azure4",[250, 170, 200, 100])
            pygame.draw.rect(win, "azure4",[10, 300, 200, 100])
            pygame.draw.rect(win, "azure4",[250, 300, 200, 100])
            win.blit(FONT_LOGIN.render("Nivel 1", True, "black"), (400, 0))
            win.blit(FONT_PERGUNTA.render(self.perguntas_lv1[self.lv1_index], True, "black"), (20, 40))
            if self.perguntas_lv1[self.lv1_index] == "7 // 2 vale quanto?":
                win.blit(FONT_PERGUNTA.render("3.5", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("3", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("Vai dar erro de compilação", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("4", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    acertos += 1
                    streak += 1
                    while troca_ok == False:
                        self.lv1_index = random.randint(0, len(self.perguntas_lv1) - 1)
                        if self.lv1_index != 0:
                            troca_ok = True
                            time.sleep(0.5)
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
            elif self.perguntas_lv1[self.lv1_index] == "print 'Hello, ', 'world', tera qual resultado no console?'":
                win.blit(FONT_PERGUNTA.render("Hello, world", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("Hello, ", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("Vai dar erro de compilação", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("world", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    while troca_ok == False:
                        self.lv1_index = random.randint(0, len(self.perguntas_lv1) - 1)
                        if self.lv1_index != 1:
                            troca_ok = True
                            time.sleep(0.5)
                    acertos += 1
                    streak += 1
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
            elif self.perguntas_lv1[self.lv1_index] == "10 % 2 vale quanto?":
                win.blit(FONT_PERGUNTA.render("0", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("5, ", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("0.2", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("Vai dar erro de compilação", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    acertos += 1
                    streak += 1
                    while troca_ok == False:
                        self.lv1_index = random.randint(0, len(self.perguntas_lv1) - 1)
                        if self.lv1_index != 2:
                            troca_ok = True
                            time.sleep(0.5)
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
            elif self.perguntas_lv1[self.lv1_index] == "Qual o simbolo utilizado para adicionar comentarios?":
                win.blit(FONT_PERGUNTA.render("#", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("//", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("/*   */", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("<!--   -->", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    acertos += 1
                    streak += 1
                    while troca_ok == False:
                        self.lv1_index = random.randint(0, len(self.perguntas_lv1) - 1)
                        if self.lv1_index != 3:
                            troca_ok = True
                            time.sleep(0.5)
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
            elif self.perguntas_lv1[self.lv1_index] == "100 / 0 vale quanto?":
                win.blit(FONT_PERGUNTA.render("0", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("100", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("Vai dar erro de compilação", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("False", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    while troca_ok == False:
                        self.lv1_index = random.randint(0, len(self.perguntas_lv1) - 1)
                        if self.lv1_index != 4:
                            troca_ok = True
                            time.sleep(0.5)
                    acertos += 1
                    streak += 1
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
        elif lv2_aberto:
            level = 2
            pygame.draw.rect(win, "azure4",[10, 170, 200, 100])
            pygame.draw.rect(win, "azure4",[250, 170, 200, 100])
            pygame.draw.rect(win, "azure4",[10, 300, 200, 100])
            pygame.draw.rect(win, "azure4",[250, 300, 200, 100])
            win.blit(FONT_LOGIN.render("Nivel 2", True, "black"), (400, 0))
            win.blit(FONT_PERGUNTA.render(self.perguntas_lv2[self.lv2_index], True, "black"), (20, 40))
            if self.perguntas_lv2[self.lv2_index] == "print('Hello' + 'world') terá qual resultado?":
                win.blit(FONT_PERGUNTA.render("Hello world", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("Helloworld", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("Vai dar erro de compilação", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("Hello+world", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    acertos += 1
                    streak += 1
                    while troca_ok == False:
                        self.lv2_index = random.randint(0, len(self.perguntas_lv2) - 1)
                        if self.lv2_index != 0:
                            troca_ok = True
                            time.sleep(0.5)
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
            elif self.perguntas_lv2[self.lv2_index] == "idade = 7 + 5 = 4, idade terá qual valor?":
                win.blit(FONT_PERGUNTA.render("idade = 4", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("idade = 12", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("Vai dar erro de sintaxe", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("idade = 8", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    while troca_ok == False:
                        self.lv2_index = random.randint(0, len(self.perguntas_lv2) - 1)
                        if self.lv2_index != 1:
                            troca_ok = True
                            time.sleep(0.5)
                    acertos += 1
                    streak += 1
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
            elif self.perguntas_lv2[self.lv2_index] == "7.5 // 2 vale quanto":
                win.blit(FONT_PERGUNTA.render("3", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("3.5, ", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("4", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("Vai dar erro de compilação", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    acertos += 1
                    streak += 1
                    while troca_ok == False:
                        self.lv2_index = random.randint(0, len(self.perguntas_lv2) - 1)
                        if self.lv2_index != 2:
                            troca_ok = True
                            time.sleep(0.5)
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
            elif self.perguntas_lv2[self.lv2_index] == "Como posso criar uma função em Python?":
                win.blit(FONT_PERGUNTA.render("def", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("func", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("method", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("class", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    acertos += 1
                    streak += 1
                    while troca_ok == False:
                        self.lv2_index = random.randint(0, len(self.perguntas_lv2) - 1)
                        if self.lv2_index != 3:
                            troca_ok = True
                            time.sleep(0.5)
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
            elif self.perguntas_lv2[self.lv2_index] == "Como posso contar a frequência de elementos em uma lista em Python?":
                win.blit(FONT_PERGUNTA.render("len()", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("sum()", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("count()", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("find()", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    while troca_ok == False:
                        self.lv2_index = random.randint(0, len(self.perguntas_lv2) - 1)
                        if self.lv2_index != 4:
                            troca_ok = True
                            time.sleep(0.5)
                    acertos += 1
                    streak += 1
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
        elif lv3_aberto:
            level = 3
            pygame.draw.rect(win, "azure4",[10, 170, 200, 100])
            pygame.draw.rect(win, "azure4",[250, 170, 200, 100])
            pygame.draw.rect(win, "azure4",[10, 300, 200, 100])
            pygame.draw.rect(win, "azure4",[250, 300, 200, 100])
            win.blit(FONT_LOGIN.render("Nivel 3", True, "black"), (400, 0))
            win.blit(FONT_PERGUNTA.render(self.perguntas_lv3[self.lv3_index], True, "black"), (20, 40))
            if self.perguntas_lv3[self.lv3_index] == "Como posso verificar se uma lista está vazia em Python?":
                win.blit(FONT_PERGUNTA.render("is_empty()", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("len()", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("check_empty()", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("empty()", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    acertos += 1
                    streak += 1
                    while troca_ok == False:
                        self.lv3_index = random.randint(0, len(self.perguntas_lv3) - 1)
                        if self.lv3_index != 0:
                            troca_ok = True
                            time.sleep(0.5)
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
            elif self.perguntas_lv3[self.lv3_index] == "Como posso converter uma string em maiúsculas em Python?":
                win.blit(FONT_PERGUNTA.render("uppercase()", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("convert_upper()", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("upper()", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("to_upper()", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    while troca_ok == False:
                        self.lv3_index = random.randint(0, len(self.perguntas_lv3) - 1)
                        if self.lv3_index != 1:
                            troca_ok = True
                            time.sleep(0.5)
                    acertos += 1
                    streak += 1
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
            elif self.perguntas_lv3[self.lv3_index] == "Como posso criar um dicionário vazio em Python?":
                win.blit(FONT_PERGUNTA.render("dicionario = {}", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("dicionario = dict", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("dicionario = dict()", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("dicionario = []", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    acertos += 1
                    streak += 1
                    while troca_ok == False:
                        self.lv3_index = random.randint(0, len(self.perguntas_lv3) - 1)
                        if self.lv3_index != 2:
                            troca_ok = True
                            time.sleep(0.5)
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
            elif self.perguntas_lv3[self.lv3_index] == "Como posso criar uma classe em python?":
                win.blit(FONT_PERGUNTA.render("class", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("def", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("public class", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("<nome_da_classe>", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    acertos += 1
                    streak += 1
                    while troca_ok == False:
                        self.lv3_index = random.randint(0, len(self.perguntas_lv3) - 1)
                        if self.lv3_index != 3:
                            troca_ok = True
                            time.sleep(0.5)
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
            elif self.perguntas_lv3[self.lv3_index] == "Como faço para instalar um pacote externo em Python usando o pip?":
                win.blit(FONT_PERGUNTA.render("pip install 'nomedopacote'", True, "black"), (10, 170))
                win.blit(FONT_PERGUNTA.render("python -m pip install nomedopacote", True, "black"), (250, 170))
                win.blit(FONT_PERGUNTA.render("pip install nomedopacote", True, "black"), (10, 300))
                win.blit(FONT_PERGUNTA.render("pip install nomedopacote==versao", True, "black"), (250, 300))
                if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
                elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    while troca_ok == False:
                        self.lv3_index = random.randint(0, len(self.perguntas_lv3) - 1)
                        if self.lv3_index != 4:
                            troca_ok = True
                            time.sleep(0.5)
                    acertos += 1
                    streak += 1
                elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    if shield:
                        shield = False
                        time.sleep(0.5)
                    else:
                        streak = 0
        elif lv_endless_aberto:
            win.blit(FONT_LOGIN.render("Nivel INF", True, "black"), (400, 0))
            win.blit(FONT_LOGIN.render("Gerar outra pergunta", True, "black"), (325, 425))
            pygame.draw.rect(win, "azure4",[10, 170, 200, 100])
            pygame.draw.rect(win, "azure4",[250, 170, 200, 100])
            pygame.draw.rect(win, "azure4",[10, 300, 200, 100])
            pygame.draw.rect(win, "azure4",[250, 300, 200, 100])
            # Gera uma nova pergunta
            if self.nova_pergunta.collidepoint(mpos) and pygame.mouse.get_pressed()[0] or self.pergunta_ok == False:
                self.pergunta_ok = True
                self.shuffle_ok = False
                gerar_texto_chatgpt()

            # Tratamento de dados enviado pelo chatgpt
            global completion
            pattern = r"\n|\?|a\)|b\)|c\)|d\)"
            string = completion.choices[0].message.content
            elementos = re.split(pattern, string)
            elementos = [element for element in elementos if element.strip()]
            
            # Muda de ordem as respotas para a correta não ficar sempre como primeira
            if not self.shuffle_ok:
                self.resp_certa = elementos[1]
                self.respostas.clear()
                self.respostas.append(elementos[1])
                self.respostas.append(elementos[2])
                self.respostas.append(elementos[3])
                self.respostas.append(elementos[4])
                random.shuffle(self.respostas)

                if self.resp_certa in self.respostas[0]:
                    self.correta = 1
                elif self.resp_certa in self.respostas[1]:
                    self.correta = 2
                elif self.resp_certa in self.respostas[2]:
                    self.correta = 3
                elif self.resp_certa in self.respostas[3]:
                    self.correta = 4
                
                self.shuffle_ok = True

            pergunta = elementos[0]
            win.blit(FONT_PERGUNTA.render(pergunta, True, "black"), (0, 50))
            win.blit(FONT_PERGUNTA.render(self.respostas[0], True, "black"), (10, 170))
            win.blit(FONT_PERGUNTA.render(self.respostas[1], True, "black"), (250, 170))
            win.blit(FONT_PERGUNTA.render(self.respostas[2], True, "black"), (10, 300))
            win.blit(FONT_PERGUNTA.render(self.respostas[3], True, "black"), (250, 300))
            
            if self.resp1.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                if self.correta == 1:
                    self.acerto = True
                else:
                    self.erro = True
            elif self.resp2.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                if self.correta == 2:
                    self.acerto = True
                else:
                    self.erro = True
            elif self.resp3.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                if self.correta == 3:
                    self.acerto = True
                else:
                    self.erro = True
            elif self.resp4.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                if self.correta == 4:
                    self.acerto = True
                else:
                    self.erro = True

            if self.acerto and self.erro == False:
                msg = FONT_MOEDAS.render("Acertou :)", True, "black")
                win.blit(msg, (720, 110))
                self.erro = False
                
            if self.erro and self.acerto == False:
                msg = FONT_MOEDAS.render("Errou :(", True, "black")
                win.blit(msg, (720, 110))
                self.acerto = False
            
            # Caso os dois ficassem verdadeiros por um bug, eles são definidos como False para arrumar o bug
            if self.erro == True and self.acerto == True:
                self.erro = False
                self.acerto = False
        
        if lv_aberto:
                voltar = FONT_LOGIN.render("Voltar", True, "black")
                win.blit(voltar,(400, 500))
                if voltar_rect_pergunta.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
                    self.voltar_ok = True
                    # Abre o metodo do banco de dados para poder atualizar as moedas do jogador
                    login.banco_de_dados(login.moedas, login.xp)
                    acertos = 0
                    streak = 0
                    level = 0
                    # Escolhe uma nova pergunta aleatória caso o jogador saia do nível para não ficar na mesma
                    self.lv1_index = random.randint(0, len(self.perguntas_lv1) - 1)
                    self.lv2_index = random.randint(0, len(self.perguntas_lv2) - 1)
                    self.lv3_index = random.randint(0, len(self.perguntas_lv3) - 1)
                    self.respostas_ok = False
    
class Resposta(Pergunta):
    def __init__(self):
        pass

    def calcular_pontos(self, acertos, streak, level):
        formula = acertos * 100 * level * (1 + streak / 10)
        global boost
        global boost_ok
        if boost and formula != 0:
            pontos = formula * 1.25
            boost = False
            boost_ok = False
        else:
            pontos = formula
        return pontos

    def calcular_xp(self): 
        xp_novo = self.calcular_pontos(acertos, streak, level) / 10
        return xp_novo
    
    def calcular_moedas(self):
        moedas_novo = self.calcular_pontos(acertos, streak, level) / 4
        return moedas_novo

class Login(Pergunta):
    global coins
    # Método utilizado para permitir a sobrecarga de métodos no Python
    def __init__(self):
        self.inicio = False
        self.login = False
        self.cadastro = False
        self.sair_rect = pygame.Rect(830, 0, 64, 64)
        self.cadastrar_rect = pygame.Rect(500, 300, 200, 50)
        self.login_rect = pygame.Rect(200, 300, 125, 50)
        self.usuario_rect = pygame.Rect(90, 92, 600, 40)
        self.senha_rect = pygame.Rect(90, 192, 600, 40)
        self.voltar_rect = pygame.Rect(400, 500, 100, 30)
        self.enviar_rect = pygame.Rect(400, 400, 100 , 30)
        self.entrar_rect = pygame.Rect(350, 400, 150, 50)
        self.usuario_click = False
        self.senha_click = False
        self.login_pronto = False
        self.cadastro_pronto = False
        self.entrar = True
        self.senha = ""
        self.usuario = ""
        self.pergunta = Pergunta()
        self.resposta = Resposta()
        self.moedas = coins
        self.xp = 0
    
    def mostrar_xpmoedas(self):
        xp = FONT_MOEDAS.render(str(self.xp), True, "black")
        win.blit(xp, (765, 110))
        moedas = FONT_MOEDAS.render(str(self.moedas), True, "black")
        win.blit(moedas, (765, 210))
        xp_img = pygame.image.load(os.path.join("imgs", "Xp.png"))
        win.blit(xp_img, (700, 100))
        moedas_img = pygame.image.load(os.path.join("imgs", "Coin.png"))
        win.blit(moedas_img, (700, 200))

    def banco_de_dados(self, moedas, xp):
        # Conecta no banco de dados
        try:
            with psycopg.connect(
                dbname="neondb",
                user="Parelho",
                password="ns3Nolki1RzC",
                host="ep-little-field-610508.us-east-2.aws.neon.tech",
                port= '5432'
                ) as db:
                # Abre o cursor para verificar os valores das tabelas
                with db.cursor() as cursor:

                    # Insere o cadastro no banco de dados
                    if self.cadastro_pronto == True:
                        add_usuario = "INSERT INTO Usuario VALUES(%s, %s, %s, %s);"
                        data_usuario = (self.usuario, self.senha, xp, moedas)
                        cursor.execute(add_usuario, data_usuario)
                        db.commit()
                        add_loja = "INSERT INTO Loja VALUES(%s);"
                        data_loja = (self.usuario,)
                        cursor.execute(add_loja, data_loja)
                        db.commit()
                        self.cadastro_pronto = False
                    
                    # Verifica se o usuario e senha inseridos existem no banco de dados
                    if self.login_pronto:
                        query = "SELECT * FROM Usuario"
                        cursor.execute(query)
                        rows = cursor.fetchall()
                        usuario_encontrado = False

                        for row in rows:
                            if self.usuario == row[0] and self.senha == row[1]:
                                print("Usuario encontrado")
                                self.xp = int(row[2])
                                self.moedas = int(row[3])
                                self.login_pronto = False
                                self.inicio = False
                                self.login = False
                                usuario_encontrado = True
                                break
                        else:
                            if not usuario_encontrado:
                                print("Usuario nao encontrado")
                                self.login_pronto = False

                    # Caso o usuario sai de algum nível calcula a xp e moedas novas e atualiza no banco de dados
                    if pergunta.voltar_ok:
                        global acertos
                        global level
                        global streak
                        xp_nova = int(self.xp + self.resposta.calcular_xp())
                        query = f"UPDATE usuario SET xp = '{xp_nova}' WHERE username = '{self.usuario}';"
                        cursor.execute(query)
                        self.xp = xp_nova
                        moedas_nova = int(self.moedas + self.resposta.calcular_moedas())
                        query = f"UPDATE usuario SET moedas = '{moedas_nova}' WHERE username = '{self.usuario}';"
                        cursor.execute(query)
                        self.moedas = moedas_nova
                    
                    # Desbloquea os cosmeticos e remove o custo do banco de dados
                    global cosmetico1_desbloqueado
                    if cosmetico1_desbloqueado:
                        if self.moedas < 200:
                            cosmetico1_desbloqueado = False
                        else:
                            coins_novo = self.moedas - 200
                            query = f"UPDATE usuario SET moedas = '{coins_novo}' WHERE username = '{self.usuario}';"
                            cursor.execute(query)
                            query = "UPDATE loja SET cosmetico1 = %s WHERE username = %s;"
                            data = (True, self.usuario)
                            cursor.execute(query, data)
                            self.moedas = coins_novo
                            cosmetico1_desbloqueado = False
                    
                    global cosmetico2_desbloqueado
                    if cosmetico2_desbloqueado:
                        if self.moedas < 200:
                            cosmetico2_desbloqueado = False
                        else:
                            coins_novo = self.moedas - 200
                            query = f"UPDATE usuario SET moedas = '{coins_novo}' WHERE username = '{self.usuario}';"
                            cursor.execute(query)
                            query = "UPDATE loja SET cosmetico2 = %s WHERE username = %s;"
                            data = (True, self.usuario)
                            cursor.execute(query, data)
                            self.moedas = coins_novo
                            cosmetico2_desbloqueado = False
                    
                    global cosmetico3_desbloqueado
                    if cosmetico3_desbloqueado:
                        if self.moedas < 200:
                            cosmetico3_desbloqueado = False
                        else:
                            coins_novo = self.moedas - 200
                            query = f"UPDATE usuario SET moedas = '{coins_novo}' WHERE username = '{self.usuario}';"
                            cursor.execute(query)
                            query = "UPDATE loja SET cosmetico3 = %s WHERE username = %s;"
                            data = (True, self.usuario)
                            cursor.execute(query, data)
                            self.moedas = coins_novo
                            cosmetico3_desbloqueado = False
                    
                    global cosmetico1_ok
                    global cosmetico2_ok
                    global cosmetico3_ok
                    username = self.usuario
                    consulta_cosmetico1 = "SELECT cosmetico1 FROM loja WHERE username = %s;"
                    cursor.execute(consulta_cosmetico1, (username,))
                    resultado_cosmetico1 = cursor.fetchone()

                    # Executar a consulta para o cosmetico2
                    consulta_cosmetico2 = "SELECT cosmetico2 FROM loja WHERE username = %s;"
                    cursor.execute(consulta_cosmetico2, (username,))
                    resultado_cosmetico2 = cursor.fetchone()

                    # Executar a consulta para o cosmetico3
                    consulta_cosmetico3 = "SELECT cosmetico3 FROM loja WHERE username = %s;"
                    cursor.execute(consulta_cosmetico3, (username,))
                    resultado_cosmetico3 = cursor.fetchone()

                    # Verificar o resultado do cosmetico1
                    if resultado_cosmetico1 and resultado_cosmetico1[0] is True:
                        cosmetico1_ok = True

                    # Verificar o resultado do cosmetico2
                    if resultado_cosmetico2 and resultado_cosmetico2[0] is True:
                        cosmetico2_ok = True

                    # Verificar o resultado do cosmetico3
                    if resultado_cosmetico3 and resultado_cosmetico3[0] is True:
                        cosmetico3_ok = True


                    # Disconta o valor dos powerups do jogador no banco de dados
                    global boost
                    if boost:
                        if self.moedas < 100:
                            boost = False
                        else:
                            coins_novo = self.moedas - 100
                            query = f"UPDATE usuario SET moedas = '{coins_novo}' WHERE username = '{self.usuario}';"
                            cursor.execute(query)
                            self.moedas = coins_novo
                    global shield
                    global shield_ok
                    if shield:
                        if self.moedas < 100:
                            shield = False
                        else:
                            coins_novo = self.moedas - 100
                            query = f"UPDATE usuario SET moedas = '{coins_novo}' WHERE username = '{self.usuario}';"
                            cursor.execute(query)
                            self.moedas = coins_novo
                            shield_ok = True
        except:
            print("Erro de conexao com o banco de dados")

    def fazer_login(self):
        # Mostrando os campos de usuário e senha para o jogador
        tela = FONT.render("Login", True, "black")
        win.blit(tela, (400, 10))
        pygame.draw.rect(win, "black",[90, 92, 600, 40], 2, 3)
        usuario = FONT_LOGIN.render("Usuario: ", True, "black")
        win.blit(usuario, (95, 92))
        pygame.draw.rect(win, "black",[90, 192, 600, 40], 2, 3)
        senha = FONT_LOGIN.render("Senha: ", True, "black")
        win.blit(senha, (100, 192))
        pygame.draw.rect(win, "black",[398, 498, 79, 34], 0, 3)
        pygame.draw.rect(win, "burlywood2",[400, 500, 75, 30], 0, 3)
        voltar = FONT_LOGIN.render("Voltar", True, "white")
        win.blit(voltar,(400, 500))
        pygame.draw.rect(win, "black",[398, 398, 84, 34], 0, 3)
        pygame.draw.rect(win, "burlywood2",[400, 400, 80, 30], 0, 3)
        enviar = FONT_LOGIN.render("Enviar", True, "white")
        win.blit(enviar, (400, 400))
        mpos = pygame.mouse.get_pos()

        # Checa se o mouse está em cima do texto de voltar e se o jogador clicou com o botão esquerdo do mouse
        if self.voltar_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.inicio = True
            self.login = False

        # Checa se o mouse está em cima do texto de usuário e se o jogador clicou com o botão esquerdo do mouse
        if self.usuario_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            # usuario_click utilizado para o jogador não ter que ficar segurando o botão esquerdo do mouse para poder digitar, provavelmente existe uma solução melhor
            self.usuario_click = True
        # Caso o jogador clique fora da caixa, ela deixa de aceitar input
        elif not self.usuario_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.usuario_click = False
            textinput_usuario.cursor_visible = False
        if self.usuario_click:
            # Checa todas as frames se ouve alguma mudança na string
            textinput_usuario.update(events)
        # Coloca a string na tela
        win.blit(textinput_usuario.surface, (200, 100))

        # Checa se o mouse está em cima do texto de senha e se o jogador clicou com o botão esquerdo do mouse
        if self.senha_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            # senha_click utilizado para o jogador não ter que ficar segurando o botão esquerdo do mouse para poder digitar, provavelmente existe uma solução melhor
            self.senha_click = True
        # Caso o jogador clique fora da caixa, ela deixa de aceitar input
        elif not self.senha_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.senha_click = False
            textinput_senha.cursor_visible = False
        
        if self.senha_click:
            # Checa todas as frames se ouve alguma mudança na string
            textinput_senha.update(events)
        # Coloca a string na tela
        win.blit(textinput_senha.surface, (200, 200))
        
        if self.enviar_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            time.sleep(1)
            self.login_pronto = True
            self.usuario = textinput_usuario.value
            self.senha = textinput_senha.value
            textinput_usuario.value = ""
            textinput_senha.value = ""

       # Similar ao método de login, fora a parte que está comentada
    def fazer_cadastro(self):
        tela = FONT.render("Cadastro", True, "black")
        win.blit(tela, (350, 10))
        
        pygame.draw.rect(win, "black",[90, 92, 600, 40], 2, 3)
        usuario = FONT_LOGIN.render("Usuario: ", True, "black")
        win.blit(usuario, (95, 92))
        pygame.draw.rect(win, "black",[90, 192, 600, 40], 2, 3)
        senha = FONT_LOGIN.render("Senha: ", True, "black")
        win.blit(senha, (100, 192))
        pygame.draw.rect(win, "black",[398, 498, 79, 34], 0, 3)
        pygame.draw.rect(win, "burlywood2",[400, 500, 75, 30], 0, 3)
        voltar = FONT_LOGIN.render("Voltar", True, "white")
        win.blit(voltar,(400, 500))
        pygame.draw.rect(win, "black",[398, 398, 84, 34], 0, 3)
        pygame.draw.rect(win, "burlywood2",[400, 400, 80, 30], 0, 3)
        enviar = FONT_LOGIN.render("Enviar", True, "white")
        win.blit(enviar, (400, 400))
        mpos = pygame.mouse.get_pos()

        if self.voltar_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.inicio = True
            self.cadastro = False
    
        if self.usuario_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.usuario_click = True
        elif not self.usuario_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.usuario_click = False
            textinput_usuario.cursor_visible = False
        
        if self.usuario_click:
            textinput_usuario.update(events)
        win.blit(textinput_usuario.surface, (200, 100))

        if self.senha_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.senha_click = True
        elif not self.senha_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            self.senha_click = False
            textinput_senha.cursor_visible = False
        
        if self.senha_click:
            textinput_senha.update(events)
        win.blit(textinput_senha.surface, (200, 200))

        if self.enviar_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            time.sleep(1)
            self.cadastro_pronto = True
            self.usuario = textinput_usuario.value
            self.senha = textinput_senha.value
            self.cadastro = False
            self.inicio = True
            textinput_usuario.value = ""
            textinput_senha.value = ""
    def tela_inicio(self):
        mpos = pygame.mouse.get_pos()
        sair = pygame.image.load(os.path.join("imgs", "Close.png"))
        win.blit(sair, (830, 0))
        if self.sair_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            global fechar
            fechar = True

        bem_vindo = FONT.render("Efetue seu login ou cadastro", True, "black")
        win.blit(bem_vindo, (165, 100))
        pygame.draw.rect(win, "black",[498, 298, 199, 54], 0, 3)
        pygame.draw.rect(win, "burlywood2",[500, 300, 195, 50], 0, 3)
        cadastrar = FONT.render("Cadastrar", True, "white")
        win.blit(cadastrar, (500, 300))
        pygame.draw.rect(win, "black",[198, 298, 129, 64], 0, 3)
        pygame.draw.rect(win, "burlywood2",[200, 300, 125, 60], 0, 3)
        login = FONT.render("Login", True, "white")
        win.blit(login, (200, 300))

        # Checa se o mouse está em cima do botão de cadastro
        global logoff
        if logoff:
            self.inicio = True
            logoff = False
    
        if self.cadastrar_rect.collidepoint(mpos):
            # Checa se o jogador clicou com o botão esquerdo do mouse
            if pygame.mouse.get_pressed()[0]:
                self.cadastro = True
                self.inicio = False
        # Checa se o mouse está em cima do botão de login     
        elif self.login_rect.collidepoint(mpos):
            # Checa se o jogador clicou com o botão esquerdo do mouse
            if pygame.mouse.get_pressed()[0]:
                self.login = True
                self.inicio = False
    def tela_boas_vindas(self):
        bem_vindo = FONT.render("Bem-vindo ao CodeQuiz", True, "black")
        win.blit(bem_vindo, (200, 100))

        pygame.draw.rect(win, "black",[348, 398, 154, 54], 0, 3)
        pygame.draw.rect(win, "burlywood2",[350, 400, 150, 50], 0, 3)
        entrar = FONT.render("Entrar", True, "white")
        win.blit(entrar, (360, 400))

        mpos = pygame.mouse.get_pos()
        if self.entrar_rect.collidepoint(mpos) and pygame.mouse.get_pressed()[0]:
            time.sleep(0.2)
            self.inicio = True
            self.entrar = False


# Utilizado para criar a string que será utilizada pelo pygame_textinput
textinput_usuario = pygame_textinput.TextInputVisualizer()
textinput_senha = pygame_textinput.TextInputVisualizer()

running = True
jogador = Jogador()
login = Login()
nivel = SeletorDeNivel()
pergunta = Pergunta()

while running:
    # Utilizado para ver os inputs do jogador
    events = pygame.event.get()
    # Fecha o loop caso a aba do pygame seja fechada
    for event in events:
        if event.type == pygame.QUIT or fechar == True:
            running = False
    # Coloca o tema do fundo na tela atrás de todo o resto que for desenhado
    win.fill(jogador.tema)

    if login.entrar:
        login.tela_boas_vindas()
    if logoff == True:
        login.tela_inicio()
    # Login().inicio é utilizado para ver se o a tela de boas vindas deve ser mostrada ou não
    if login.inicio:
        login.tela_inicio()
    # Se login for True, será aberta a tela de login
    elif login.login:
        login.fazer_login()
        if login.login_pronto:
            login.banco_de_dados(login.moedas, login.xp)
    # Se login for False, será aberta a tela de cadastro
    elif login.cadastro:
        login.fazer_cadastro()
        if login.cadastro_pronto:
            login.banco_de_dados(login.moedas, login.xp)
    
    elif login.inicio == False and login.login == False and login.cadastro == False and login.entrar == False:
        if jogador.opcoes_aberto == False and jogador.loja_aberta == False and nivel.lv_aberto == False:
            jogador.menu_principal()
            login.mostrar_xpmoedas()
            nivel.selecionar_nivel(login.xp)
        elif jogador.opcoes_aberto:
            jogador.opcoes()
        elif jogador.loja_aberta:
            jogador.loja()
            if boost == True and login.moedas >= 100 and boost_ok == False:
                login.banco_de_dados(login.moedas, login.xp)
                boost_ok = True
            elif shield == True and login.moedas >= 100 and shield_ok == False:
                login.banco_de_dados(login.moedas, login.xp)
            elif cosmetico1_desbloqueado == True and login.moedas >= 200:
                login.banco_de_dados(login.moedas, login.xp)
            elif cosmetico2_desbloqueado == True and login.moedas >= 200:
                login.banco_de_dados(login.moedas, login.xp)
            elif cosmetico3_desbloqueado == True and login.moedas >= 200:
                login.banco_de_dados(login.moedas, login.xp)
        elif nivel.lv_aberto:
            pergunta.nivel(nivel.lv1_aberto, nivel.lv2_aberto, nivel.lv3_aberto, nivel.lv_endless_aberto , nivel.voltar_rect_pergunta, nivel.lv_aberto)
            if pergunta.voltar_ok:
                nivel.lv_aberto = False
                pergunta.voltar_ok = False
                time.sleep(0.5)


    # Da update nos métodos do pygame
    pygame.display.update()
    # Taxa de FPS, atualmente está 30 FPS
    clock.tick(30)
# Para as operações do pygame para garantir que o código vai terminar de ser executado
pygame.quit()
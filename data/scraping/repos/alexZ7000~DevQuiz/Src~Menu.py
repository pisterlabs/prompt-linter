from tkinter import *
from tkinter.messagebox import *
import smtplib
import email.message
import pygame.mixer_music
import os
from dotenv import load_dotenv
import openai
from time import sleep
import ConexaoComBancoDeDados as bd
from chatgpt import criar_pergunta
from chatgpt import responder_pergunta

root = Tk()  # Janela
root.title("DevQuiz")  # Título da janela
x, y = (root.winfo_screenwidth()), (root.winfo_screenheight())  # Pega a resolução do monitor sem considerar a escala
root.geometry(f'{x}x{y}')  # Dimensão da tela
root.minsize(1910, 1070)  # Resolução mínima para redimencionalizar
root.maxsize(1920, 1080)  # Forçar resolução Full HD (1920x1080) do DevQuiz
root.attributes("-fullscreen", 1)  # Colocar o DevQuiz em tela cheia
frame = Frame(root)


class Menu:
    """Classe que irá definir todos os itens que todos os "Menu" vão usar"""

    def __init__(self):
        """Criação da minha tela"""
        self.theme_txt = None
        self.dev_system = None
        self.escuro = None
        self.claro = None
        self.frame = Frame(root, bg="#ccccff")
        self._build_screen()

    c = 0

    def change_theme(self):
        """ Função para mudar de tema "Claro" para tema "Escuro"
        :param self: Menu
        :returns: Não retorna nada"""
        self.claro = "#ccccff"
        self.escuro = "#1D1D66"
        if self.c % 2 == 0:
            self.frame.config(bg=self.claro)
            return "#ccccff"
        else:
            self.frame.config(bg=self.escuro)
            return "#1D1D66"

    def set_dev_system(self, dev_system):
        """ Função para colocar os objetos referenciados no "DevSystem" em todas as Classes que herdarem de "Menu".
            :param dev_system: Pegar referencias
            :returns: Não retorna nada
        """
        self.dev_system = dev_system

    def show(self):
        """ Função para mostrar todos os widgets que forem "self.frame" """
        self.frame.pack(fill=BOTH, expand=True)

    def hide(self):
        """ Função para esconder widgets que não 
        serão mais usados em uma tela nova e para 
        excluir caracteres inseridos nos "Entry" """
        self.frame.forget()
        self.reset_entry()

    def _build_screen(self):
        """Função para construir minha tela, mas eu não preciso
        construir nenhuma tela em menu, essa função deve ser ignorada"""
        pass

    def reset_entry(self):
        """Função para limpar os caracteres inseridos no "Entry" """
        pass

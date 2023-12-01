from django.contrib import admin
from django.urls import path
from webapp.views import openai
from webapp.views import autenticacao

urlpatterns = [    
    path("correcao", openai.correcao, name="correcao"),
    path("criacao", openai.criacao, name="criacao"),
    path("geral", openai.geral, name="geral"),
    path("historico", openai.historico, name="historico"),
    path("deletar_registro/<id_do_registro>", openai.deletar_registro, name="deletar_registro"),
    path("signin", autenticacao.signin, name="signin"),
    path("signout", autenticacao.signout, name="signout"),
    path("registro", autenticacao.signup, name="registro"),
]

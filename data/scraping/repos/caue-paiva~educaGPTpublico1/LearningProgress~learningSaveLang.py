import openai
import json
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

Userinput="""(Enem/2013) É preciso ressaltar que, de todas as capitanias brasileiras, Minas era a mais urbanizada. Não havia ali hegemonia de um ou dois grandes centros. A região era repleta de vilas e arraiais, grandes e pequenos, em cujas ruas muita gente circulava.

PAIVA, E. F. O ouro e as transformações na sociedade colonial. São Paulo: Atual, 1998.

As regiões da América portuguesa tiveram distintas lógicas de ocupação. Uma explicação para a especificidade da região descrita no texto está identificada na:

a) apropriação cultural diante das influências externas.
b) produção manufatureira diante do exclusivo comercial.
c) insubordinação religiosa diante da hierarquia eclesiástica.
d) fiscalização estatal diante das particularidades econômicas.
e) autonomia administrativa diante das instituições metropolitanas.
 
Essa é a resposta verdadeira sempre: Alternativa correta é d) fiscalização estatal diante das particularidades econômicas.

aluno: a resposta certa é a)

"""

#primeiro numero é o total de questões e o segundo o número de acertos;
Progresso_materias = {
    "Historia_Brasil": [0, 0],
    "Historia_antiga": [0, 0],
    "Idade_media": [0, 0]
}

Sub_Topicos_Globais =["Historia_Brasil","Historia_antiga","Idade_media"]

def Salva_progresso(Subtopico, resposta_correta, Resposta_aluno):
  "Identifica o subTopico da questão e se o aluno acertou a questão ou não"
  Progresso_materias[Subtopico][0] += 1
  if (resposta_correta == Resposta_aluno):
   Progresso_materias[Subtopico][1] += 1

  return f"progresso salvo no subtopico {Subtopico}"

def Mostra_progresso(Subtopico):
  totalQuest = Progresso_materias[Subtopico][0]
  Acertos = Progresso_materias[Subtopico][1]
  porce_acertos = ((Acertos/totalQuest)*100)
  resposta = f"No Subtopico {Subtopico}, você fez {totalQuest} questões e teve {porce_acertos}% de acertos "
  print(resposta)
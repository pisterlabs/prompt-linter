from django.shortcuts import render
import requests
from .models import *
from report.models import *
import openai
from Iris import settings
import openai
import subprocess
from report.ask_openai import *

openai.api_key = settings.openai.api_key

import openai
import re



def Audio_chat_view(request):
    audio = Audio.objects.filter(user_id=request.user.id).last()
    texte = TextGen.objects.filter(Audio=audio).last()
    text = texte.text
    answers = {}
    response=""
    if request.method == "POST":
        query = request.POST.get('answer')
        if query != " ":
        
            # response = ask_questions(text,query,1500,"davinci")
            response = exec(text,query)
            if response ==" ":
                response="Désolé Le texte que vous avez fournie ne me permet pas de répondre à cette question"
            gpt = response
            req = Request.objects.create(query=query, answer=gpt,filename=audio.filename)
            req.save()
            answers[query]= gpt
            # answers = AddValueToDict(query, answers, "response"+query, type( "response"))
    requ = Request.objects.filter(filename=audio.filename)
    try:
        seq = len(requ) - 1
        r = requ[:seq]
    except:
        r= requ
    context = {
        "report":audio,
        "answers":answers,
        "requ":r,
    }
    return render(request, 'window.html',context )

def Audio_chat_view_old(request,id):
    audio = Audio.objects.get(user_id=request.user.id, id=id)
    texte = TextGen.objects.filter(Audio=audio).last()
    text = texte.text
    answers = {}
    response=""
    if request.method == "POST":
        query = request.POST.get('answer')
        if query != " ":
        
            # response = ask_questions(text,query,1500,"davinci")
            response = exec(text,query)
            if response ==" ":
                response="Désolé Le texte que vous avez fournie ne me permet pas de répondre à cette question"
            gpt = response
            req = Request.objects.create(query=query, answer=gpt,filename=audio.filename)
            req.save()
            answers[query]= gpt
            # answers = AddValueToDict(query, answers, "response"+query, type( "response"))
    requ = Request.objects.filter(filename=audio.filename)
    try:
        seq = len(requ) - 1
        r = requ[:seq]
    except:
        r= requ
    context = {
        "report":audio,
        "answers":answers,
        "requ":r,
    }
    return render(request, 'window.html',context )


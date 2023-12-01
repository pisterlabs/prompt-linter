import os
from django.shortcuts import render
from .serializers import *
from .models import Bio
from rest_framework.generics import GenericAPIView
from django.http import JsonResponse,HttpRequest, HttpResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
import openai
import json
from rest_framework import serializers



# API KEY

# OPENAI_API_KEY = ''
# openai.api_key=OPENAI_API_KEY  

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Create your views here.

def bio_list(request):
    bios = Bio.objects.get()
    serializer= BioSerializer(bios)
    return Response(serializer.data)


def toString(s):  
    str1 = " "
    return (str1.join(s))

def get_key(val, operations):
    for key,value in operations.items():
        if val==value:
            print(key)
            return key



class Solve(GenericAPIView):
    serializer_class = SolveSerializer

    def post(self,request,*args,**kwargs):
        header = {
            "Access-Control-Allow-Origin":"*"
        }
        data = {
                "operation_type":self.request.data['operation_type'],
                "x":self.request.data['x'],
                "y":self.request.data['y'],
            }
        operations = {
            'addition': '+',
            'subtraction': '-',
            'multiplication': '*',
            'division':'/'
        }    
        print(type(operations))
        opr = data['operation_type'].lower()
        x = int(data['x'])
        y = int(data['y'])
        real_opr = ""
        result = ""
        if opr in operations:
            temp_opr = operations[opr]
            real_opr = get_key(temp_opr, operations)
            result = eval(f'{x}{temp_opr}{y}')
        else:

            # varieties = "Find Synonyms:\n\n Subtraction | addition |multiplication"
            varieties = f'Find Math Operator: \n \n {opr} '

            response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=varieties,
            temperature=0.2,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
            )
            responses = toString(response['choices'][0]['text'].split(','))
            for special_op in responses:
                for op in operations:
                    if operations[op] == special_op:
                        result = eval(f'{x}{operations[op]}{y}')  
                        temp_opr = operations[op]
                        real_opr = get_key(temp_opr, operations)

        return  JsonResponse({
                    "slackUsername":"timmy-spark",
                    "result": result,
                    "operation_type":real_opr
                    })







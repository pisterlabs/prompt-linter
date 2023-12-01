import os
import openai
from rest_framework.views import APIView
from rest_framework import permissions,status
from rest_framework.response import Response
from django.db.models import Q

from inventoryAPI.models import *

openai.organization = "org-HrXKRQTivn0etgp22bq3pYJ7"
openai.api_key = "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"

class ReceipeSuggestionListAPI(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self,request,pk=None):

        try:
            inv = Inventory.objects.filter(Q(user_associated=request.user.id) | Q(sharedTo__id=request.user.id)).get(pk=pk)
        except Exception as e:
            print(e)
            return Response({'message':'something went wrong'},status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        prdls = InventoryProduct.objects.filter(invAssociated=inv.pk)
        prd = ''
        for i in prdls:
            prd  += f'{i.prodAssociated.prdName} ,'


        prompt = f"List of Food receipe items in format of 'food name (Cooking time: time required to made food)' that can be made with {prd} along with their cooking time are"
        response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=1, max_tokens=2048)

        msg = response['choices'][0]['text']
        smsg = msg.split('\n')
        tp = "(Cooking Time: "
        res = []

        for i in smsg:
            print(i)
            if len(i) < 4:
                continue
            op = 0

            for j in range(len(i)):
                if i[j].isalpha():
                    op = j
                    break

            for j in range(len(i)):
                if i[j:j+len(tp)] == tp:
                    res.append({"food_item":i[op:j-1],"duration":i[j+len(tp):-2]})

        print(res)
        return Response(res,status=status.HTTP_200_OK)

class ReceipeSuggestionAPi(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self,request,format=None):

        receipeName = request.data['receipeName']
        prompt = f"Provide me step by step instructions and nutrition value with ingredients of {receipeName} are"
        response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=1, max_tokens=2048)

        print(openai.Model.list())


        return Response({'msg':response['choices'][0]['text']},status=status.HTTP_200_OK)
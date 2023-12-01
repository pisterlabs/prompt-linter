from bdb import GENERATOR_AND_COROUTINE_FLAGS
from inspect import GEN_RUNNING
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from sumarizer.settings import WOMBOHEADER
from .models import *
import requests
import json
import time
from coHere import *
from .serializers import *
import json
# Create your views here.

class RetrieveArticle(APIView):
    
    def post(self, request):
        data = request.data
        
        for i in data.keys():
            article = ArticleModel(title = data[i]['title'],
                                  session = data[i]['session'],
                                  date = data[i]['date'],
                                  first = data[i]['1'],
                                  second = data[i]['2'],
                                  third = data[i]['3'],
                                  fourth = data[i]['4'],
                                  url = data[i]['url'])
            
            style_id = 23
            imgspec = self.image_spec(0.1, 1560, 1560) 
            inputspec = self.input_spec(style_id, article.title, imgspec)  
            article.photo = self.get_wombo(inputspec)
            article.save()
            
        return Response({"succes"}, status = 200)
    
    
    def get(self, request):
        response_list = []
        first = int(request.query_params['first'])
        last = int(request.query_params['last'])
       
        for i in range(first, last + 1):
            summary_list = []
            article = ArticleModel.objects.get(pk = i)
            first = article.first
            second = article.second
            third = article.third
            fourth = article.fourth
            articledata = ArticleSerializer(article).data
            summary_list.append(GenerateSummary(first))
            summary_list.append(GenerateSummary(second))
            summary_list.append(GenerateSummary(third))
            summary_list.append(GenerateSummary(fourth))
            articledata['summary'] = summary_list
            response_list.append(articledata)
       
           
        return Response(response_list, status = 200)
    
    def get_wombo(self, spec):
        post_payload = json.dumps({
            "use_target_image": False
        })
        
        post_response = requests.request(
            "POST", "https://api.luan.tools/api/tasks/",
            headers = WOMBOHEADER, data = post_payload
        )
        print(post_response.json())
        task_id = post_response.json()['id']
        task_id_url = f"https://api.luan.tools/api/tasks/{task_id}"
        put_payload = json.dumps(
            spec
        )
        
        requests.request("PUT", task_id_url, headers = WOMBOHEADER, data = put_payload)
        while True:            
            response_json = requests.request(                    
                "GET", task_id_url, headers=WOMBOHEADER).json()     
            
            state = response_json["state"]    
        
            if state == "completed":                    
                r = requests.request("GET", response_json["result"])                    
                with open("image.jpg", "wb") as image_file:                            
                    image_file.write(r.content)                        
                print("image saved successfully :)")                    
                break 
            
            elif state =="failed":                    
                print("generation failed :(")                    
                break            
            time.sleep(3)
        
        
        
    def input_spec(self, style_id, prompt, spec = None):
        data = {
            'style': style_id,
            'prompt': prompt
        }
        
        if spec:
            for x in spec.keys():
                data[x] = spec[x]    
        
        return {
            'input_spec': data
        }
            
        
        
    def image_spec(self, weight, width, height):
        return {
            'target_image_weight': weight,
            'width': width,
            'height': height
        }
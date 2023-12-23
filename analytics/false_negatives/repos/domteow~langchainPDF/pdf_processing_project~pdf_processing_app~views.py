import base64
import json
import os
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt

from .models import UploadedPDF

from .forms import UploadPDFForm

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

import tempfile

# Create your views here.

@csrf_exempt
def upload_pdf(request):
    if request.method == 'POST':
        
            form = UploadPDFForm(request.POST, request.FILES)
            print(form)
            print(request.FILES)
            # if form.is_valid():
            temp_dir = tempfile.mkdtemp()

            
            temp_pdf_file = request.FILES.get('pdf_file')
        
            temp_file_path = os.path.join(temp_dir, temp_pdf_file.name)
            with open(temp_file_path, 'wb') as temp_file:
                for chunk in temp_pdf_file.chunks():
                    temp_file.write(chunk)
            

            loader = PyPDFLoader(temp_file_path)


            docs = loader.load()

            # temp_pdf_file.close()

            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
            chain = load_summarize_chain(llm, chain_type="stuff")

            chain.run(docs)

            # Define prompt
            prompt_template = """Write a detailed summary of the following:
            "{text}"
            CONCISE SUMMARY:"""
            prompt = PromptTemplate.from_template(prompt_template)

            # Define LLM chain
            llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo-16k")
            llm_chain = LLMChain(llm=llm, prompt=prompt)

            # Define StuffDocumentsChain
            stuff_chain = StuffDocumentsChain(
                llm_chain=llm_chain, document_variable_name="text"
            )

            docs = loader.load()
            processed_result = stuff_chain.run(docs)

            response_data = {
                'summary': processed_result 
            }

            print(response_data)

            return JsonResponse(response_data, status=200)
            
            # else:
            #     print("hi")
            #     print(form.errors)
            #     return JsonResponse({'error': 'Invalid form data'})
            
        

    else:
        
        return HttpResponse("Method not allowed", status=405)  

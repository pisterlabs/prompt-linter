from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from .forms import upload_file_form, topic_extraction_form
from .models import Topic_extraction_Documents
import boto3, botocore
from botocore.config import Config
import time, os, sys
from smart_open import open
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import gensim
import gensim.corpora as corpora
#from gensim.utils import simple_preprocess
#from gensim.models import CoherenceModel
from gensim.test.utils import datapath
from gensim import models
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm as tqdm
from pprint import pprint
import json
from .tasks import extract_topics, topic_test_task
from celery import Celery
from celery.result import AsyncResult
from django_project.celery import app


@login_required
def topic_import_data_type(request):
    return render(request, 'topic_extraction/topic_import_data_type.html')


@login_required
def topic_preview_data(request):
    docs = Topic_extraction_Documents.objects.filter(author=request.user.id)
    return render(request, 'topic_extraction/topic_preview_data.html', {'docs':docs})


@login_required
def topic_csv_preview_file(request):
    docs = Topic_extraction_Documents.objects.filter(author=request.user.id)
    return render(request, 'topic_extraction/topic_csv_preview_file.html', {'docs':docs})

@login_required
def topic_json_preview_file(request):
    docs = Topic_extraction_Documents.objects.filter(author=request.user.id)
    return render(request, 'topic_extraction/topic_json_preview_file.html', {'docs':docs})

@login_required
def topic_results_page(request):
	if request.method == 'POST':
		result_id = request.session['result']# define result
		result = AsyncResult(result_id, app=app)# get the result of the task
		json_data = result.get()
		# print(data) # testing
		pk = request.session['pk']
		doc = Topic_extraction_Documents.objects.get(pk=pk)# get the document ref from the database
		documentName = str(doc.document)# get the real name of the doc

		aws_id = settings.AWS_ACCESS_KEY_ID# AWS ACCESS
		aws_secret = settings.AWS_SECRET_ACCESS_KEY
		REGION = 'eu-west-1'

		client = boto3.client('s3', region_name = REGION, aws_access_key_id=aws_id,
		        aws_secret_access_key=aws_secret)

		bucket_name = settings.AWS_STORAGE_BUCKET_NAME

		object_key = documentName
		csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
		body = csv_obj['Body']
		csv_string = body.read().decode('utf-8')

		data = pd.read_csv(StringIO(csv_string)) #CREATE DATAFRAME FROM CSV

		documents = data['content'] # assign docs to content
		return render(request, 'topic_extraction/topic_results_page.html', {'json_data':json_data, 'documents':documents})
	else:
		docs = Topic_extraction_Documents.objects.filter(author=request.user.id, document__contains=".csv")
		return render(request, 'topic_extraction/topic_display_csv_text.html', {'docs':docs})

    


@login_required
def topic_delete_docs(request, pk):
    if request.method == 'POST':
        doc = Topic_extraction_Documents.objects.get(pk=pk)
        doc.delete()
    return redirect('topic_extraction-topic_preview_data')


@login_required
def topic_select_doc(request, pk):
    if request.method == 'POST':# check for post request
        doc = Topic_extraction_Documents.objects.get(pk=pk)# get the document ref from the database
        documentName = str(doc.document)# get the real name of the doc
        ext = check_file(request, pk, documentName)
        if ext == '.other': # check if doc is unsupported format
            messages.error(request, f'Please use a file format such as .txt, .csv json or begin text extraction process on a new file')
            return redirect('topic_extraction-topic_preview_file')
        else:
            return redirect('topic_extraction-topic_display_text')
    else:
        messages.error(request, f'unable to process file')
    return render(request, 'topic_extraction-topic_preview_file.html')
    
@login_required
def topic_json_upload_file(request):
    if request.method == 'POST':
        form = upload_file_form(request.POST, request.FILES)
        form.instance.author = request.user
        if form.is_valid():
            form.save()
            return redirect('topic_extraction-topic_json_preview_file')
    else:
        form = upload_file_form()
    return render(request, 'topic_extraction/topic_json_upload_file.html', {'form': form})


@login_required
def topic_csv_upload_file(request):
    if request.method == 'POST':
        form = upload_file_form(request.POST, request.FILES)
        form.instance.author = request.user
        if form.is_valid():
            form.save()
            return redirect('topic_extraction-topic_csv_preview_file')
    else:
        form = upload_file_form()
    return render(request, 'topic_extraction/topic_csv_upload_file.html', {'form': form})


@login_required
def topic_display_csv_text(request):
    form = topic_extraction_form()
    docs = Topic_extraction_Documents.objects.filter(author=request.user.id, document__contains="csv")
    return render(request, 'topic_extraction/topic_display_csv_text.html', {'docs':docs, 'form':form})

@login_required
def topic_display_json_text(request):
    docs = Topic_extraction_Documents.objects.filter(author=request.user.id, document__contains="json")
    return render(request, 'topic_extraction/topic_display_extracted_json.html', {'docs':docs})
    
@login_required # require login to be activated
def topic_extraction(request):
	if request.method == 'POST':
		try:
			choices = [5,10,15,20,25,30,35,40,45,50]
			form = topic_extraction_form(request.POST)

			if form.is_valid():
				choice =form.cleaned_data['No_of_topics'] # gets the choice field entered
				choice = int(choice[0])-1
				topics = choices[choice]

			pk = request.session['pk']
			result = extract_topics.delay(pk, topics) # send to celery worker for topic ectraction

			request.session['result'] = result.id # get the id of the task for retrival from storage
			#docs = Topic_extraction_Documents.objects.filter(author=request.user.id, document__contains=".csv")
			return render(request, 'topic_extraction/topic_progress_csv.html', context={'task_id': result.task_id})
		except:
			messages.error(request, f'unable to process file')
			docs = Topic_extraction_Documents.objects.filter(author=request.user.id, document__contains=".csv")
			return render(request, 'topic_extraction/topic_display_csv_text.html')
	else:
		docs = Topic_extraction_Documents.objects.filter(author=request.user.id, document__contains=".csv")
		return render(request, 'topic_extraction/topic_display_csv_text.html')

	

def topic_read_csv_doc(request, pk):
    if request.method == 'POST':# check for post request
        request.session['pk'] = pk        
        doc = Topic_extraction_Documents.objects.get(pk=pk)# get the document ref from the database
        documentName = str(doc.document)# get the real name of the doc     
        aws_id = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret = os.environ.get('AWS_SECRET_ACCESS_KEY')
        REGION = 'eu-west-1'
        client = boto3.client('s3', region_name = REGION, aws_access_key_id=aws_id,
                aws_secret_access_key=aws_secret)
        bucket_name = "doc-sort-file-upload"
        object_key = documentName
        csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
        body = csv_obj['Body']
        csv_string = body.read().decode('utf-8')
        data = pd.read_csv(StringIO(csv_string))
        content = data.head().to_dict()
        #pprint(content)# testing
        request.session['content'] = content
        
        return redirect('topic_extraction-topic_display_csv_text')
    else:
        messages.error(request, f'unable to read c.s.v. file!')
    return render(request, 'topic_extraction-topic_csv_preview_file.html')

def topic_read_json_doc(request, pk):
    if request.method == 'POST':# check for post request
        request.session['pk'] = pk      
        doc = Topic_extraction_Documents.objects.get(pk=pk)# get the document ref from the database
        documentName = str(doc.document)# get the real name of the doc       
        aws_id = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret = os.environ.get('AWS_SECRET_ACCESS_KEY')
        REGION = 'eu-west-1'
        client = boto3.client('s3', region_name = REGION, aws_access_key_id=aws_id,
                aws_secret_access_key=aws_secret)
        bucket_name = "doc-sort-file-upload"
        object_key = documentName
        csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
        body = csv_obj['Body']
        csv_string = body.read().decode('utf-8')
        data = pd.read_csv(StringIO(csv_string))
        content = data.head().to_dict()
        pprint(content)
        request.session['content'] = content
        return redirect('topic_extraction-topic_display_json_text')
    else:
        messages.error(request, f'unable to read json file!')
    return render(request, 'topic_extraction-topic_json_preview_file.html')
  

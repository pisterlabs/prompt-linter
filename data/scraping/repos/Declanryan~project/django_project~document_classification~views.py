from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.contrib.auth.decorators import login_required
from .forms import tag_selection_form, model_name_selection_form, upload_file_form, topic_extraction_form
from .models import Classification_Documents
import boto3, botocore
from botocore.config import Config
import time, os, sys
from tempfile import NamedTemporaryFile
from smart_open import open
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import datapath
from gensim import models
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm as tqdm
from pprint import pprint
import json



@login_required
def import_data_type(request):
    return render(request, 'document_classification/import_data_type.html')


@login_required
def preview_data(request):
    docs = Classification_Documents.objects.filter(author=request.user.id)
    return render(request, 'document_classification/preview_data.html', {'docs':docs})


@login_required
def csv_preview_file(request):
    docs = Classification_Documents.objects.filter(author=request.user.id)
    return render(request, 'document_classification/csv_preview_file.html', {'docs':docs})

@login_required
def json_preview_file(request):
    docs = Classification_Documents.objects.filter(author=request.user.id)
    return render(request, 'document_classification/json_preview_file.html', {'docs':docs})


@login_required
def tag_selection(request):
    if request.method == 'POST':
        labels_list =[]
        form = tag_selection_form(request.POST)
        if form.is_valid():
            labels_list.extend('Tag0, Tag1, Tag2, Tag3, Tag4, Tag5, Tag6, Tag7, Tag8, Tag9')
            return redirect('document_classification-results_page')
    else:
        form = tag_selection_form()
        return render(request, 'document_classification/tag_selection.html', {'form': form})

@login_required
def delete_docs(request, pk):
    if request.method == 'POST':
        doc = Classification_Documents.objects.get(pk=pk)
        doc.delete()
    return redirect('document_classification-preview_data')


@login_required
def select_doc(request, pk):
    if request.method == 'POST':# check for post request
        doc = Classification_Documents.objects.get(pk=pk)# get the document ref from the database
        documentName = str(doc.document)# get the real name of the doc
        ext = check_file(request, pk, documentName)
        if ext == '.other': # check if doc is unsupported format
            messages.error(request, f'Please use an extracted file format such as .txt, .csv or begin extraction process on a new file')
            return redirect('document_classification-csv_preview_file')
        else:
            return redirect('document_classification-display_csv_text')
    else:
        messages.error(request, f'unable to process file')
    return render(request, 'document_classification-csv_preview_file.html')
    
@login_required
def json_upload_file(request):
    if request.method == 'POST':
        form = upload_file_form(request.POST, request.FILES)
        form.instance.author = request.user
        if form.is_valid():
            form.save()
            return redirect('document_classification-json_preview_file')
    else:
        form = upload_file_form()
    return render(request, 'document_classification/json_upload_file.html', {'form': form})


@login_required
def csv_upload_file(request):
    if request.method == 'POST':
        form = upload_file_form(request.POST, request.FILES)
        form.instance.author = request.user
        if form.is_valid():
            form.save()
            return redirect('document_classification-csv_preview_file')
    else:
        form = upload_file_form()
    return render(request, 'document_classification/csv_upload_file.html', {'form': form})


@login_required
def upload_confirmation(request):
    if request.method == 'POST':
        form = upload_file_form(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            return redirect('document_classification-preview_data')
    else:
        form = upload_file_form()
    return render(request, 'document_classification/upload_confirmation.html', {'form': form})

@login_required
def display_csv_text(request):
    form = topic_extraction_form()
    docs = Classification_Documents.objects.filter(author=request.user.id, document__contains="csv")
    return render(request, 'document_classification/display_csv_text.html', {'docs':docs, 'form':form})

@login_required
def display_json_text(request):
    docs = Classification_Documents.objects.filter(author=request.user.id, document__contains="json")
    return render(request, 'document_classification/display_extracted_json.html', {'docs':docs})
    

def read_csv_doc(request, pk):
    if request.method == 'POST':# check for post request
        request.session['pk'] = pk        
        doc = Classification_Documents.objects.get(pk=pk)# get the document ref from the database
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
        
        return redirect('document_classification-display_csv_text')
    else:
        messages.error(request, f'unable to read c.s.v. file!')
    return render(request, 'document_classification-csv_preview_file.html')

def read_json_doc(request, pk):
    if request.method == 'POST':# check for post request
        request.session['pk'] = pk      
        doc = Classification_Documents.objects.get(pk=pk)# get the document ref from the database
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
        return redirect('document_classification-display_json_text')
    else:
        messages.error(request, f'unable to read json file!')
    return render(request, 'document_classification-json_preview_file.html')
  
@login_required
def results_page(request):
    return render(request, 'document_classification/results_page.html')
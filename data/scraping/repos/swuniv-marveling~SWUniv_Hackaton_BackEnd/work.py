from dotenv import load_dotenv
from flask import Flask, jsonify, request, abort
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from bson import ObjectId
from werkzeug.utils import secure_filename
from google.cloud import storage
import requests
from pymongo import MongoClient
import tempfile
import requests
import openai
import os
import uuid
import io
from bson.objectid import ObjectId

load_dotenv()

client = MongoClient('mongodb://loca:loca23!@127.0.0.1', 27017)

def bucket_filename(user_id, filename):
    return str(user_id) + '/' + str(uuid.uuid4()) + '_' + secure_filename(filename)

def upload_file(bucket, user_id, file):
    filename = bucket_filename(user_id, file.filename)
    blob = bucket.blob(filename)
    blob.upload_from_string(file.read(), content_type=file.content_type, timeout=300)
    return blob.public_url

def upload_output_from_url(bucket, user_id, fileurl):
    response = requests.get(fileurl)
    image_file = response.content

    filename = bucket_filename(user_id, "output.png")
    blob = bucket.blob(filename)
    blob.upload_from_string(image_file, content_type="image/png", timeout=300)
    return blob.public_url

def _create_work(request):
    user_id = get_jwt_identity()
    input_image = request.files['input']
    mask_image = request.files['mask']
    prompt = request.form['prompt']

    storage_client = storage.Client()
    bucket = storage_client.bucket(os.getenv('bucket_name'))

    if input_image and mask_image:
        openai.api_key = os.getenv('OPENAI_KEY')
        input_public_url = upload_file(bucket, user_id, input_image)
        mask_public_url = upload_file(bucket, user_id, mask_image)
        input_image.seek(0)
        mask_image.seek(0)
        response = openai.Image.create_edit(
            image=io.BufferedReader(input_image),
            mask=io.BufferedReader(mask_image),
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
    else:
        abort(400)
    
    db = client.userinfo

    work_info = {
        'user_id': user_id,
        'input_url': input_public_url,
        'mask_url': mask_public_url,
        'output_url': upload_output_from_url(bucket, user_id, response['data'][0]['url']),
        'prompt_text': prompt,
    }
    
    
    try:
        result = db.work.insert_one(work_info)
    except Exception as e:
        print(e)
        return abort(401)

    return jsonify({'success':1, 'id': str(result.inserted_id)})

def _get_work_list():
    result = {}
    work_list = []
    db = client.userinfo
    collection = db['work']
    
    user_id = get_jwt_identity()
    print(user_id)
    
    keys = ['input_url', 'output_url', 'mask_url', 'prompt_text']
    
    # select all requests results of the user
    try:
        for doc in collection.find():
            try:
                if doc['user_id'] == user_id:
                    #print("equal!!")
                    new_doc = {key: doc[key] for key in keys if key in doc}
                    #print(new_doc)
                    new_doc['work_id'] = str(doc['_id'])
                    work_list.append(new_doc)
                    #print(work_list)
            except Exception as innere:
                print(innere)
                print(doc)
                continue
    except Exception as e:
        print(e)
        result['success'] = 0

    result['work_list'] = work_list
    return result

def _get_work(work_id):
    result = {}
    db = client.userinfo
    collection = db['work']
    work_id = ObjectId(work_id)
    
    user_id = get_jwt_identity()
    print(user_id)
    
    keys = ['input_url', 'output_url', 'mask_url', 'prompt_text']

    try:
        doc = collection.find_one({'_id': work_id})
        if doc['user_id'] == user_id:
            result['success'] = 1
            new_doc = {key: doc[key] for key in keys if key in doc}
            new_doc['work_id'] = str(doc['_id'])
            result['work'] = new_doc
        else:
            raise Exception("Invalid token")
    except Exception as e:
        result['success'] = 0
    return result

def _delete_work(work_id):
    result = {}
    db = client.userinfo
    collection = db['work']
    work_id = ObjectId(work_id)

    user_id = get_jwt_identity()
    print(user_id)

    try:
        doc = collection.find_one({'_id': work_id})
        if doc['user_id'] == user_id:
            print(collection.delete_one({'_id': work_id}))
            result['success'] = 1
        else:
            raise Exception("Invalid work id")

    except Exception as e:
        result['success'] = 0
    return jsonify(result)

def _get_all_work():
    result = {}
    work_list = []
    user_id = get_jwt_identity()    # 해당 user id는 보여지는 것에서 제외
    db = client.userinfo
    collection = db['work']

    keys = ['input_url', 'output_url', 'mask_url', 'prompt_text']

    for doc in collection.find():
        try:
            if doc['user_id'] != user_id:
                new_doc = {key: doc[key] for key in keys if key in doc}
                new_doc['work_id'] = str(doc['_id'])
                work_list.append(new_doc)
        except Exception as e:
            print(doc)
            print(e)
            continue
    
    work_list.reverse()    # show the recent images
    result['work_list'] = work_list
    
    return result

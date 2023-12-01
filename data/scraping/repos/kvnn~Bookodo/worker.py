import base64
import boto3
import os
import time
from uuid import uuid4

from celery import Celery
import openai
from openai.error import InvalidRequestError
import redis

from config import settings

from sql_app.database import SessionLocal
from sql_app.models import List


redis_url = settings.redis_url
redis_client = redis.from_url(redis_url)

celery = Celery(__name__)
celery.conf.broker_url = settings.celery_broker_url
celery.conf.result_backend = settings.celery_result_backend

openai.api_key = settings.openai_api_key

if settings.in_cloud:
    s3 = boto3.client('s3')
    s3_bucket_name = settings.s3_bucket_name_media
else:
    s3 = object()

@celery.task(name="generate_list_image", bind=True)
def generate_list_image(
    self,
    list_id,
    images_path,
):
    try:
        db = SessionLocal(expire_on_commit=False)
        list = db.query(List).filter(List.id == list_id).first()
        
        if not list:
            raise Exception(f'List with id {list_id} not found')

        img_prompt = f'Please generate a captivating, unique image that combines the concepts and aesthetics of the following books: {list.book_titles}'
        
        try:
            print(f'img_prompt={img_prompt}')
            
            self.update_state(
                state='PROGRESS',
                meta={'progress':'calling image_response'}
            )

            image_response = openai.Image.create(prompt=img_prompt, n=1, size="256x256", response_format='b64_json')
        except InvalidRequestError as err :
            print(f'[error] image_response failed for with error {err}')
            return {
                'state': 'FAILED',
                'error': str(err),
                'type': 'list_image_generate',       
            }

        self.update_state(
            state='PROGRESS',
            meta={'progress':'finished image_response'}
        )
        
        print(f'image_response={image_response}')

        b64s = [base64.b64decode(obj['b64_json']) for obj in image_response['data']]

        for idx, b64_img in enumerate(b64s):
            filename = f'{list.id}-{uuid4()}.png'

            if settings.in_cloud:
                filename = f'img/{filename}'
                s3.put_object(Body=b64_img, Bucket=s3_bucket_name, Key=filename, ContentType='image/png')
                # TODO: idk how necessary this is here, and it should be a config
                img_path = f"https://{s3_bucket_name}.s3.amazonaws.com/{filename}"
                print(f's3 img_path={img_path}')
            else:
                img_path = f'{images_path}/{filename}'
                directory = os.path.dirname(img_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                with open(img_path, "wb") as img_file:
                    img_file.write(b64_img)
                print(f'local img_path={img_path}')

            print(f'saving {filename} to list {list.id}')
            list.image_filename = filename
            db.commit()

            meta = {
                'type': 'list_image_generate',
            }
            self.update_state(
                state='PROGRESS',
                meta=meta
            )

        meta = {
            'type': 'list_image_generate',
            'list_id': list_id
        }

        print(f'generate_list_image_task done. returning {meta}')

        return meta

    except Exception as err:
        print (f'[error] generate_list_image failed with error {err}')
        return {
            'error': str(err)
        }
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
from sql_app.crud import (
    create_scene,
    create_scene_image,
    create_scene_image_prompt
)
from sql_app.database import SessionLocal
from sql_app.seed_values import scene_prompt_title_separator, scene_prompt_format, scene_prompt_max_length


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

@celery.task(name="generate_scene_image", bind=True)
def generate_scene_image(
    self,
    chunk_id,
    scene_id,
    scene_title,
    scene_content,
    images_path,
    aesthetic_title,
):
    try:
        generated_images = []
        img_prompt_prompt = (
            f'You are an expert generative-art prompt engineer. For the following scene, please provide a prompt that has a maximum length '
            f'of 900 characters and that will generate a fitting image for the following scene: {scene_content}. '
            f'Keep in mind that it is very important that we avoid '
            f'triggering the "safety system" exception (e.g. "Your request was rejected as a result of our safety system").'
        )
        
        img_prompt_prompt = img_prompt_prompt  # dall-e requires < 1000 chars
        # TODO: we could stream this to the UI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": img_prompt_prompt}],
            stream=True
        )
        self.update_state(
            state='PROGRESS',
            meta={'progress':'finished img_prompt_prompt'}
        )

        img_prompt = ''
        for chunk in response:
            new = chunk['choices'][0]['delta'].get('content')
            if new:
                img_prompt += new
                meta = {
                    'img_prompt': img_prompt
                }
                self.update_state(
                    state='PROGRESS',
                    meta=meta
                )

        if len(img_prompt) > 940:
            img_prompt = img_prompt[:940]
        
        img_prompt = f'{img_prompt}, {aesthetic_title} style'

        db = SessionLocal(expire_on_commit=False)
        new_img_prompt = create_scene_image_prompt(
            db=db,
            scene_id=scene_id,
            content=img_prompt
        )

        try:
            print(f'img_prompt={img_prompt}')
            
            self.update_state(
                state='PROGRESS',
                meta={'progress':'calling image_response'}
            )

            image_response = openai.Image.create(prompt=img_prompt, n=2, size="512x512", response_format='b64_json')
        except InvalidRequestError as err :
            # TODO: cycle through a new prompt
            print(f'[error] image_response failed for with error {err}')
            return {
                'state': 'FAILED',
                'error': str(err),
                'type': 'scene_image_generate',       
            }

        self.update_state(
            state='PROGRESS',
            meta={'progress':'finished image_response'}
        )

        b64s = [base64.b64decode(obj['b64_json']) for obj in image_response['data']]

        for idx, b64_img in enumerate(b64s):
            filename = f'{scene_id}-{uuid4()}.png'

            if settings.in_cloud:
                filename = f'img/{filename}'
                s3.put_object(Body=b64_img, Bucket=s3_bucket_name, Key=filename, ContentType='image/png')
                # TODO: idk how necessary this is, and it should be a config
                img_path = f"https://{s3_bucket_name}.s3.amazonaws.com/{filename}"
                print(f's3 img_path={img_path}')
            else:
                img_path = f'{images_path}/{filename}'
                with open(img_path, "wb") as img_file:
                    img_file.write(b64_img)
                print(f'local img_path={img_path}')

            db = SessionLocal(expire_on_commit=False)
            new_img = create_scene_image(
                db,
                chunk_id=chunk_id,
                scene_id=scene_id,
                scene_image_prompt_id=new_img_prompt.id,
                filename=filename
            )
            generated_images.append({
                'id': new_img.id,
                'scene_image_prompt_id': new_img_prompt.id,
                'image_path': img_path
            })
            meta = {
                'type': 'scene_image_generate',
                'images': generated_images
            }
            self.update_state(
                state='PROGRESS',
                meta=meta
            )

        meta = {
            'type': 'scene_image_generate',
            'images': generated_images,
            'chunk_id': chunk_id,
            'scene_id': scene_id
        }

        print(f'generate_scene_task done. returning {meta}')

        return meta

    except Exception as err:
        print (f'[error] generate_scene failed with error {err}')
        return {
            'error': str(err)
        }


@celery.task(name="generate_scene", bind=True)
def generate_scene(
    self,
    images_path,
    prompt,
    chunk_content,
    aesthetic_title,
    aesthetic_id,
    chunk_id,
    scene_prompt_id
):
    try:
        print(f'generate_scene_task for {prompt}')

        prompt += f'\n It is important to keep your output to a maximum of {scene_prompt_max_length} letters. '
        prompt += f'\n Also generate a concise, captivating title. '
        prompt += f'\n Please return the content in this format: {scene_prompt_format}. '
        prompt += f'\n Here is the content: {chunk_content}'

        print(f'scene_prompt={prompt}')
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        content = ''
        for chunk in response:
            new = chunk['choices'][0]['delta'].get('content')
            if new:
                content += new
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'new': new,
                        'type': 'scene_generate',
                        'content': content,
                        'progress': (len(content) / scene_prompt_max_length) if scene_prompt_max_length else 'unknown'
                    }
                )
                print(f'{new}')

        self.update_state(
            state='PROGRESS',
            meta={
                'new': new,
                'type': 'scene_generate',
                'content': content,
                'progress': (len(content) / scene_prompt_max_length) if scene_prompt_max_length else 'unknown'
            }
        )

        print(f'scene_content={content}')

        try:
            title, content = content.split(scene_prompt_title_separator)
        except ValueError as e:
            title = content[:50]
            print(f'[error] ValueError for {prompt[:50]}... with error {e}')
        
        title = title[:50]  # just in case it's too long

        # Save to DB
        db = SessionLocal(expire_on_commit=False)
        new_scene = create_scene(
            db = db,
            title = title,
            content=content,
            aesthetic_id=aesthetic_id,
            chunk_id=chunk_id,
            scene_prompt_id=scene_prompt_id
        )

        # Generaet Imagery
        scene_image_task = generate_scene_image.delay(
            chunk_id=chunk_id,
            scene_id=new_scene.id,
            scene_title=title,
            scene_content=content,
            images_path=images_path,
            aesthetic_title=aesthetic_title,
        )
        scene_image_task_id = scene_image_task.id

        print(f'called scene_image_task, scene_image_task.id={scene_image_task_id}')

        redis_client.set(scene_image_task_id, 'init')
        
        return {
            'scene_id': new_scene.id,
            'scene_image_task_id': scene_image_task_id,
            'new': new,
            'type': 'scene_generate',
            'content': content,
            'progress': (len(content) / scene_prompt_max_length) if scene_prompt_max_length else 'unknown'
        }
    except Exception as err:
        print (f'[error] generate_scene failed for with error {err}')
        return {
            'error': str(err)
        }


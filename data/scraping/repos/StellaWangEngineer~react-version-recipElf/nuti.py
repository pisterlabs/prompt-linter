
# from flask import Flask
# from celery import Celery
# import openai

# app = Flask(__name__)
# app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
# app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
# celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# celery.conf.update(app.config)

# openai.api_key = "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"

# @celery.task
# def generate_prompt(prompt):
#     model_engine = "text-davinci-002"
#     completions = openai.Completion.create(
#         engine=model_engine,
#         prompt=prompt,
#         max_tokens=50,
#         n=1,
#         stop=None,
#         temperature=0.5,
#     )
#     return completions.choices[0].text.strip()
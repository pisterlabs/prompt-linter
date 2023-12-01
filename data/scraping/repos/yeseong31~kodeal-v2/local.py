import dj_database_url
import openai

from .base import *

openai.api_key = env.get_value('OPENAI_CODEX_KEY')
openai.Engine.list()


# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env.get_value('SECRET_KEY')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Database
# https://docs.djangoproject.com/en/4.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': env.get_value('LOCAL_DB_NAME'),
        'USER': env.get_value('LOCAL_DB_USER'),
        'PASSWORD': env.get_value('LOCAL_DB_PASSWORD'),
        'HOST': '127.0.0.1',
        'PORT': '3306',
        'OPTIONS': {
            'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
            'charset': 'utf8mb4',
            'use_unicode': True,
        },
    }
}

db_from_env = dj_database_url.config(conn_max_age=500)
DATABASES['default'].update(db_from_env)

# Email Authentication
EMAIL_HOST_USER = env.get_value('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = env.get_value('EMAIL_HOST_PASSWORD')

# Generate JWT Token
JWT_SECRET_KEY = env.get_value('JWT_SECRET_KEY')

# Naver Papago
CLIENT_ID = env.get_value('CLIENT_ID')
CLIENT_SECRET = env.get_value('CLIENT_SECRET')

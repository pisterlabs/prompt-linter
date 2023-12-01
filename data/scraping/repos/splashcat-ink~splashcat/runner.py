import os
import sys

if os.environ.get('FLY_MACHINE_ID'):
    sys.path.insert(0, '/code/')

import django
from django.conf import settings
from openai import OpenAI

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'splashcat.settings')
django.setup()

from battles.models import Battle, BattleGroup
from assistant.data import upload_user_battles_to_openai, upload_battle_to_openai, upload_battle_group_to_openai
from assistant.models import Thread

client = OpenAI(api_key=settings.OPENAI_API_KEY)

thread_id = int(os.environ.get("TASK_THREAD_ID"))

thread = Thread.objects.get(pk=thread_id)

if thread.status == thread.Status.CREATED:
    print("thread already started.")
    sys.exit()

initial_prompt = thread.initial_message

if thread.content_type.model_class() == Battle:
    openai_file = upload_battle_to_openai(thread.content_object)
elif thread.content_type.model_class() == BattleGroup:
    openai_file = upload_battle_group_to_openai(thread.content_object)
else:
    openai_file = upload_user_battles_to_openai(thread.creator)

thread.openai_file_id = openai_file.id
thread.status = thread.Status.CREATED
thread.save()

client.beta.threads.messages.create(
    thread_id=thread.openai_thread_id,
    role='user',
    content=thread.initial_message,
    file_ids=[openai_file.id]
)

client.beta.threads.runs.create(
    thread_id=thread.openai_thread_id,
    assistant_id=settings.OPENAI_ASSISTANT_ID,
)

sys.exit()

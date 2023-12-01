import os
import time

from celery import Celery

from app.api.models import AiChatTask

from app.config import get_settings

import openai

# ----------------------------------------------------------------------------------------------

from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

# ----------------------------------------------------------------------------------------------

celery_app = Celery(__name__)
celery_app.conf.update(
    broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379"),
    result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379"),
    task_serializer='pickle',
    result_serializer='pickle',
    accept_content=['json','pickle']
)
# ----------------------------------------------------------------------------------------------

@celery_app.task(name="create_task")
def create_task(task_type):
    time.sleep(int(task_type) * 10)
    return True

# ----------------------------------------------------------------------------------------------

openai.api_key = get_settings().OPENAI_API_KEY

# ----------------------------------------------------------------------------------------------

@celery_app.task(name="OpenAI_Comm")
def OpenAI_Comm( aichat_task: AiChatTask ) -> AiChatTask:
    
    logger.info("OpenAI_Comm: we are in!")
    logger.info(f"OpenAI_Comm: aichat_task.aichatid '{aichat_task.aichatid}'")
    logger.info(f"OpenAI_Comm: aichat_task.status '{aichat_task.status}'")
    logger.info(f"OpenAI_Comm: aichat_task.taskid '{aichat_task.taskid}'")
    logger.info(f"OpenAI_Comm: aichat_task.prompt '{aichat_task.prompt}'")
    logger.info(f"OpenAI_Comm: aichat_task.prompt '{aichat_task.prePrompt}'")
    
    # prePrompt = '''You are a bilingual English and Spanish attorney and CA Law Professor. 
    # You work for the Gloria Martinez Law Group, a Sacramento Immigration Law firm. 
    # You are meeting a potential client whom is seeking law advice. 
    # You want them to hire the firm. 
    # You only answer truthfully. 
    # Let's work out how to help the client in a step by step way so we are sure 
    # to have the right answer, the client understands, and they hire us.
    # If multiple options exist for the client, explain each option. 
    # If you do not have an answer, say "I do not know".
    # 
    # '''
    prePrompt = aichat_task.prePrompt
    
    aiResponse = None
    if aichat_task.model=="text-davinci-003":        
        aiResponse = openai.Completion.create(model=aichat_task.model, 
                                              prompt= prePrompt + " \n" + aichat_task.prompt,
                                              temperature=0,
                                              max_tokens=900,
                                              top_p=1,
                                              frequency_penalty=0.0,
                                              presence_penalty=0.0,
                                        )["choices"][0]["text"].strip(" \n")
    
    elif aichat_task.model=="gpt-3.5-turbo" or aichat_task.model=="gpt-4":
        aiResponse = openai.ChatCompletion.create( model=aichat_task.model,
                                                   messages=[ {"role": "system", "content": prePrompt },
                                                              {"role": "user", "content": aichat_task.prompt } ],
                                                   temperature=0,
                                                   max_tokens=900,
                                                   top_p=1,
                                                   frequency_penalty=0.0,
                                                   presence_penalty=0.0,
                                                )['choices'][0]['message']['content']
    
    # communication with OpenAI complete, save results:
    aichat_task.reply = aiResponse 
    aichat_task.status = 'ready'
       
    logger.info(f"OpenAI_Comm: reply '{aichat_task.reply}'")
    
    # task is done:
    return aichat_task

# ----------------------------------------------------------------------------------------------

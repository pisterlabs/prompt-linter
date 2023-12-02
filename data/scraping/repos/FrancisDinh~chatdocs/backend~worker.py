import os
import openai
from dotenv import load_dotenv
from celery import Celery

from lib import multithreadCrawler, elastic, pdf

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

celery = Celery('worker')
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379")


@celery.task(name="processFilePdf")
def processFilePdf(agent_id, file_path):
    pdf.processFilePDF(agent_id, file_path)
    return True


@celery.task(name="crawlerUrl")
def crawlerUrl(agent_id, url, selector):
    craler = multithreadCrawler.MultiThreadedCrawler(agent_id, url, selector)
    craler.run_web_crawler()
    craler.info()
    return True
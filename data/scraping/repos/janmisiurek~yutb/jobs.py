
from rq.decorators import job
from worker import conn
from rq import Queue
from rq.decorators import job
from redis import Redis
from models import User
from youtube_utils import download_audio
from openai_utils import transcript, generate_notes, generate_social_media_content

redis_conn = Redis()
q = Queue(connection=conn)

# Helper function combines the download, transcript, and create_notes functions for asynchronous execution as an RQ job
@job('default', connection=conn, timeout=7200)
def download_transcribe_generate_notes(url, tempo, content_types, user):
    record_id = download_audio(url, tempo, user)
    transcript(record_id, user)
    generate_notes(record_id)

    if content_types:
        q.enqueue(generate_social_media_content, record_id, content_types)
    
    return record_id

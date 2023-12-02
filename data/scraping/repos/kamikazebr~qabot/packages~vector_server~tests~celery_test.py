from logger.embedding_logger import logger
from tasks import celery

from utils.constants import OPENAI_API_KEY, DB_CONNECTION_STR, DB_GUILD

# t = vector_task.dummy.delay()
logger.debug('Starting vector store update in celery')
# logger.debug(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
# logger.debug(f"DB_CONNECTION_STR: {DB_CONNECTION_STR}")
logger.debug(f"DB_GUILD: {DB_GUILD}")

t = celery.vector_store_update.delay('random-session', OPENAI_API_KEY,DB_CONNECTION_STR,DB_GUILD)
logger.debug(f"Result: {t}")
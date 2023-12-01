from src.clients.openai import langchain_processor
from src.repositories.file_indexes_repository import *
from src.repositories.open_ai_process_repository import *
from src.configs.redis_config import redis_instance
import json
import asyncio
import datetime
from src.repositories.api_key_repository import ApiKeyRepository


class SearchRequestedHandler():

    @staticmethod
    def listen():
        redisClient = redis_instance()
        pubsub = redisClient.pubsub()
        pubsub.subscribe('gpt_search')
        handler = SearchRequestedHandler()
        for message in pubsub.listen():
            try:
                item = json.loads(message.get('data'))
                asyncio.run(handler.run(item, item['process_uuid']))
            except Exception as e:
                print("An exception occurred, payload: ", message)
                print(e)

    async def run(self, data, process_uuid):
        repository = OpenAiProcessRepository()
        openai_api_key = ApiKeyRepository().get_by_organization_id(data["organization"])['api_key']
        history = None
        
        # if chat_id is present, get the chat history to use as context
        if 'chat_id' in data and data['chat_id'] is not None:
            limit = int(os.environ.get('LANGCHAIN_HISTORY_LENGTH', '3'))
            history = repository.get_chat_history(data['chat_id'], limit)

        # search the answer                        
        gpt_result = langchain_processor.search(data["q"], data["organization"], openai_api_key, history)

        # update the mongodb entity with the answer
        repository.insert({
            'process_uuid': process_uuid,
            'chat_id': data['chat_id'],
            'question': data["q"],
            'organization': data['organization'],
            'response': gpt_result,
            'created_at': datetime.datetime.utcnow(),
        })

        return gpt_result

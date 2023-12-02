import openai
from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.mongodb import MongoDBChatMessageHistory
from config import *
from helper import *

# DB functions
async def change_active_conversation(userid, conversation_name):
    id_conversation_name = get_id_conversation(userid, conversation_name)
    user_conversation_list.update_one(
        filter={'user_id': userid}, 
        update={'$set': {'active_conversation': id_conversation_name}}, 
        upsert=True)

async def store_in_user_conversation_list(user_id, username, conversation_name):
    id_conversation_name = get_id_conversation(user_id, conversation_name)
    user_conversation_list.update_one(
        filter={'user_id': user_id, 'username': username}, 
        update={
            "$set": {
                "active_conversation": id_conversation_name
            }, 
            '$addToSet': {
                'conversations': id_conversation_name
            }
        }, 
        upsert=True)

async def get_user_conversation_list(user_id):
    return user_conversation_list.find_one({'user_id': user_id})

async def delete_conversation(user_id, name):
    id_conversation_name = get_id_conversation(user_id, name)
    user_conversation_list.update_one(
        filter={'user_id': user_id},
        update={'$pull': {'conversations': id_conversation_name}},
        upsert=True
    )
    user_conversation_list.update_one(
        filter={'user_id': user_id, 'active_conversation': id_conversation_name},
        update={'$set': {'active_conversation': ''}},
        upsert=True
    )
    conversations.delete_many({'SessionId': id_conversation_name})


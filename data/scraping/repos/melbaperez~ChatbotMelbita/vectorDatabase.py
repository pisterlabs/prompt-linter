import os
import openai
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from utils import readYaml

configurationFile = readYaml("data/configuration.yaml")


class VectorDatabase():
    ''' Class to manage the vector database.'''

    def __init__(self):
        ''' Initialize the class with the configuration file. '''
        openai.organization = os.environ["OPENAI_ORGANIZATION_KEY"] 
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.OPENAI_ENGINE = configurationFile['OPENAI']['OPENAI_ENGINE']

        self.DIMENSION = configurationFile['DATABASE']['DIMENSION']
        self.COLLECTION_NAME = configurationFile['DATABASE']['COLLECTION_NAME']
        self.INDEX_PARAM = configurationFile['DATABASE']['INDEX_PARAM']
        self.QUERY_PARAM = configurationFile['DATABASE']['QUERY_PARAM']
        self.HOST = configurationFile['DATABASE']['HOST']
        self.PORT = configurationFile['DATABASE']['PORT']
        connections.connect("default", host= self.HOST, port= self.PORT)
        self.COLLECTION = Collection(self.COLLECTION_NAME)
        self.NUM_MENSAJES = 12


    def createCollection(self):
        ''' Create a collection from scratch. '''
        connections.connect("default", host= self.HOST, port= self.PORT)
        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name='flagInfo', dtype=DataType.INT64),
            FieldSchema(name='date', dtype=DataType.INT64),
            FieldSchema(name='chatId', dtype=DataType.INT64),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=self.DIMENSION)
            ]
        schema = CollectionSchema(fields)
        self.COLLECTION = Collection(name=self.COLLECTION_NAME, schema=schema)
        self.COLLECTION.create_index(field_name="embedding", index_params=self.INDEX_PARAM)
        self.COLLECTION.load()


    def createEmbed(self, text):
        ''' Create embedding vectors from text and returns them in a list. '''
        embeddings = openai.Embedding.create(
            input=text,
            engine=self.OPENAI_ENGINE
        )
        return [x['embedding'] for x in embeddings['data']]


    def insertData(self, text, flagInformation, date, chatId):
        ''' Insert data into the collection. '''
        res = self.COLLECTION.query(
            expr = f"date=={date}", 
            output_fields = ["id"]
            )
        if len(res) == 0:
            data = [[text], [flagInformation], [date], [chatId], self.createEmbed([text])]
            self.COLLECTION.insert(data)
            self.updateLastMessages(chatId)


    def searchInformation(self, text, chatId, date, topK = 8):
        '''
            Search the sentences that are commands, contains information or are statements and they must be older than 
            the first message of the conversation messages to be retrieved.
        '''
        results = ""
        listResults = []
        res = self.COLLECTION.search(self.createEmbed(text), anns_field='embedding', param=self.QUERY_PARAM, limit = topK, output_fields=['id', 'text', 'date'], 
                                     consistency_level="Strong", expr=f"flagInfo == 1 and chatId == {chatId} and date < {date}")
        for hit in res:
            for hits in hit:
                listResults += [{'text': hits.entity.get('text'), 'date': hits.entity.get('date')}]

        listResultsSorted = sorted(listResults, key=lambda k: k['date'])
        results = '\n'.join([f"- {message['text'][8:]}" for message in listResultsSorted])

        return results


    def getLastMessages(self, chatId):
        ''' Fetches the latest messages from the conversation in the database. '''
        conversation = ""
        messages = self.COLLECTION.query(
                expr = f"chatId == {chatId}", 
                output_fields = ["text", "date"],
                consistency_level="Strong"
            )
        
        sortedMessages = sorted(messages, key=lambda k: k['date'])[-self.NUM_MENSAJES:]

        for message in sortedMessages:
            conversation += message['text'] + "\n"

        conversation = conversation + "AI: "
        dateFirstMessage = sortedMessages[0]['date']

        return [conversation, dateFirstMessage]
    

    def isNewUser(self, chatId):
        ''' Checks if the user is new or not. '''
        messages = self.COLLECTION.query(
                expr = f"chatId == {chatId}", 
                output_fields = ["id"],
                consistency_level="Strong"
            )
        
        if len(messages) > 0:
            return False
        else:
            return True


    def deleteListIds(self, listId):
        ''' Delete a list of IDs from the database. '''
        expr = f"id in {listId}"
        self.COLLECTION.delete(expr)


    def getConnection(self):
        ''' Returns the connection to the database. '''
        return self.COLLECTION


    def updateLastMessages(self, chatId):
        ''' Delete the oldest messages from the database. '''
        messages = self.COLLECTION.query(
                expr = f"chatId == {chatId}", 
                output_fields = ["date", "id", "flagInfo"],
                consistency_level="Strong"
            )
        if len(messages) > self.NUM_MENSAJES+1:
            listIdMessagesToDelete = []
            sortedMessages = sorted(messages, key=lambda k: k['date'])
            for message in sortedMessages[1:-self.NUM_MENSAJES]:
                if message['flagInfo'] == 0:
                    listIdMessagesToDelete.append(message['id'])
            self.deleteListIds(listIdMessagesToDelete)


    def getChatIds(self):
        ''' Get all unique chat IDs of the users who have interacted with the chatbot.'''
        chatIds = self.COLLECTION.query(
                expr = "id > 0",
                output_fields = ["chatId"],
                consistency_level="Strong"
            )
        uniqueChatIds = list(set(chatId['chatId'] for chatId in chatIds))
        return uniqueChatIds
    
    
    def deleteUserMessages(self, chatId):
        ''' Delete all the messages of a user'''
        messages = self.COLLECTION.query(
                expr = f"chatId == {chatId}",
                output_fields = ["id"],
                consistency_level="Strong"
            )
        listMessages = list(message['id'] for message in messages)
        self.deleteListIds(listMessages)

#print(len(VectorDatabase().getChatIds()))
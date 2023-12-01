from app.main.dataAccess.queryDao import QueryDao
from app.main.dataAccess.logDao import LogDao
from app.main.dataAccess.userDao import UserDao
from app.main.service.textOperationService import TextOperationService
from app.main.service.userService import UserService
from app.main.utils.prompt import Prompt
from pymongo import MongoClient
from app.main.utils.mongoDBHandler import MongoDBHandler
from redis import Redis
import tiktoken
import openai

class DependencyInjection:
    def __initDB(self, url="mongodb://localhost:27017", database="gptxe"):
        client = MongoClient(url)
        return client[database] 

    def __initCache(self, host = "localhost", port = 6379, db = 0):
        return Redis(host, port, db)
    
    def __initQueryDao(self, app):
        return QueryDao(self.getCache(app))
    
    def __initLogDao(self, app):
        return LogDao(self.getDB(app))

    def __initUserDao(self, app):
        db = self.getDB(app)
        return UserDao(db)
    
    def __initPrompt(self):
        encoding = None
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return Prompt(encoding)
    
    def __initTextOperationService(self, app):
        return TextOperationService(self.getQueryDao(app), self.getPrompt(app), self.getChatCompletion(), app.config.get("MAX_TOKEN"))
    
    def __initUserService(self, app):
        return UserService(self.getUserDao(app))
    
    def __initMongoDBHandler(self, app):
        return MongoDBHandler(self.getLogDao(app))
    
    def getDB(self, app):
        databaseInstance = getattr(app, '_database', None)
        if databaseInstance is None:
            databaseInstance = app._database = self.__initDB(app.config.get("DATASOURCE_URI"), app.config.get("DATASOURCE_DATABASE"))
        return databaseInstance

    def getCache(self, app):
        redisInstance = getattr(app, '_cache', None)
        if redisInstance is None:
            redisInstance = app._cache = self.__initCache(app.config.get("REDIS_HOST"), app.config.get("REDIS_PORT"), app.config.get("REDIS_DB"))
        return redisInstance

    def getQueryDao(self, app):
        queryDao = getattr(app, '_queryDao', None)
        if queryDao is None:
            queryDao = app._queryDao = self.__initQueryDao(app)
        return queryDao

    def getLogDao(self, app):
        logDao = getattr(app, '_logDao', None)
        if logDao is None:
            logDao = app._logDao = self.__initLogDao(app)
        return logDao

    def getUserDao(self, app):
        userDao = getattr(app, '_userDao', None)
        if userDao is None:
            userDao = app._userDao = self.__initUserDao(app)
        return userDao
    
    def getPrompt(self, app):
        prompt = getattr(app, '_prompt', None)
        if prompt is None:
            prompt = app._prompt = self.__initPrompt()
        return prompt
    
    def getChatCompletion(self):
        return openai.ChatCompletion()
    
    def getTextOperationService(self, app):
        textOperationService = getattr(app, '_textOperationService', None)
        if textOperationService is None:
            textOperationService = app._textOperationService = self.__initTextOperationService(app)
        return textOperationService
    
    def getUserService(self, app):
        userService = getattr(app, '_userService', None)
        if userService is None:
            userService = app._userService = self.__initUserService(app)
        return userService
    
    def getMongoDBHandler(self, app):
        mongoDBHandler = getattr(app, '_mongoDBHandler', None)
        if mongoDBHandler is None:
            mongoDBHandler = app._mongoDBHandler = self.__initMongoDBHandler(app)
        return mongoDBHandler
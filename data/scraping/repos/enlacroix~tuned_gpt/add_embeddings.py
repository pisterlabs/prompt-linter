import time
from typing import Literal

import pandas as pd
import openai
import configparser


class EmbeddingConverter:
    def __init__(self,
                 filePath: str,
                 maxAttemptsToGetResponse: int = 3,
                 fileFormat: Literal['csv', 'xlsx'] = 'csv',
                 handledColumnName: str = 'Question'):
        self.filePath = filePath
        # fileFormat = filePath[-4:]
        self.dataframe: pd.DataFrame = pd.read_csv(filePath) if fileFormat == 'csv' else pd.read_excel(filePath)
        self.maxAttemptsToGetResponse = maxAttemptsToGetResponse
        self.dataframe = self.dataframe[~self.dataframe[handledColumnName].isna()]
        # Оставим только вопросы (разрешить и другие символы?)
        self.dataframe = self.dataframe[self.dataframe[handledColumnName].str.contains('\?')]
        self.handledColumnName = handledColumnName
        openai.api_key = EmbeddingConverter._getKey()

    @staticmethod
    def _getKey():
        config = configparser.ConfigParser()
        config.read('config.ini')
        return config['passwords']['openAIKey']
        # self.tgToken = config['passwords']['tgToken']

    @staticmethod
    def getEmbedding(text, model: str = 'text-embedding-ada-002'):
        text = text.replace("\n", " ")
        print(text)
        return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

    def createEmbedDatabase(self, model: str = 'text-embedding-ada-002'):
        self.dataframe['ada_embedding'] = self.dataframe[self.handledColumnName].apply(EmbeddingConverter.processRow)
        self.dataframe.to_csv(self.filePath[:-4] + '_EMBED.csv', index=False)

    @staticmethod
    def processRow(row):
        x = EmbeddingConverter.getEmbedding(row)
        time.sleep(25)
        return x


# n = EmbeddingConverter(filePath='subgraphs_faq.xlsx', fileFormat='xlsx')
# n.createEmbedDatabase()


from pandasai import PandasAI
from pandasai.llm import OpenAI
import os
from pandas import DataFrame

class VpiDatabase:
    
    def __init__(self) -> None:
        model = OpenAI(os.getenv('OPENAI_API_KEY'))
        self._pandas = PandasAI(model)
    
    def get_query(self, df:DataFrame, question: str):
        pass
    
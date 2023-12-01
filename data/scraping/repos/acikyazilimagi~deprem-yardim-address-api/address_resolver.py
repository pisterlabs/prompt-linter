# Applications
from helpers.google_geocode_api import GoogleGeocodeAPI
from helpers.regex_api import ExtractInfo
from helpers.ner_api import NerApi
from helpers.openai_api import OpenAI_API

import pandas as pd
import joblib

class AddressAPI:
    def __init__(self, GOOGLE_API_KEY, OPENAI_API_KEY, NER_API_KEY):
        self.google_api = GoogleGeocodeAPI(GOOGLE_API_KEY)

        sehir_data = pd.read_csv("helpers/data/il_ilce_v3.csv")
        kp_dict = joblib.load("helpers/data/sehir_kp_objs.joblib")
        sehir_dict = joblib.load("helpers/data/sehir_dict.joblib")
        self.regex_api = ExtractInfo(kp_dict, sehir_dict, sehir_data)

        self.ner_url = "https://api-inference.huggingface.co/models/deprem-ml/deprem-ner"
        self.ner_api = NerApi(self.ner_url, NER_API_KEY)
        
        self.open_api = OpenAI_API(OPENAI_API_KEY)

    def google_geocode_api_request(self, address_text: str, entry_id: int):
        result = self.google_api.request(address_text)
        result['id'] = entry_id
        return result

    def regex_api_request(self, address_text: str, entry_id: int):
        result = self.regex_api.extract(address_text)
        result['id'] = entry_id
        return result

    def ner_api_request(self, address_text: str, entry_id: int):
        result = self.ner_api.query(address_text)
        result['id'] = entry_id
        return result
    
    def openai_api_request(self, address_text: str, entry_id: int):
        result = self.open_api.single_request(address_text)
        result['id'] = entry_id
        return result
import json
import io
import sys
from googlesearch import search
import json
import requests
from bs4 import BeautifulSoup
import openai
import  requests
import openai
import json
import requests
import urllib
import json
import pandas as pd
from requests_html import HTML
from requests_html import HTMLSession
import requests
import copy

class NoneDecoder(json.JSONDecoder):
    def decode(self, s, *args, **kwargs):
        result = super(NoneDecoder, self).decode(s, *args, **kwargs)
        return self.replace_text_none_with_none(result)

    def replace_text_none_with_none(self, value):
        if isinstance(value, list):
            return [self.replace_text_none_with_none(item) for item in value]
        elif isinstance(value, dict):
            return {k: self.replace_text_none_with_none(v) for k, v in value.items()}
        elif value == "None":
            return None
        elif value == "null":
            return None
        else:
            return self.clean_text(value)

    def clean_text(self, text):
        #if type(text) == type("string"):
        #    cleaned_text = text.replace('\n', '').replace('\r', '').strip()
        #    return cleaned_text
        return text


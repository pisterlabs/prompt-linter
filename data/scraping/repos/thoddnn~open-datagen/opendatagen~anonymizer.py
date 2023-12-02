import re
import spacy
from opendatagen.model import OpenAIChatModel, ModelName
from opendatagen.utils import load_file

class Anonymizer:

    NER_PLACEHOLDER = {
        "PERSON": "{person}",
        "ORG": "{organization}",
        "GPE": "{location}",
        "DATE": "{date}",
        "TIME": "{time}",
        "NORP": "{group}",
        "FAC": "{facility}",
        "LOC": "{location}",
        "PRODUCT": "{product}",
        "EVENT": "{event}",
        "WORK_OF_ART": "{artwork}",
        "LAW": "{law}",
        "LANGUAGE": "{language}",
        "MONEY": "{money}",
        "PERCENT": "{percentage}",
        "ORDINAL": "{ordinal}",
        "CARDINAL": "{number}",
        # Add more if needed
    }

    REGEX_PATTERN = {
        "{phone_number}": r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
        "{email}": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "{credit_card_pattern}": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}",
        "{address_pattern}": r"\d{1,5}\s\w+(\s\w+)*,\s\w+,\s\w+(\s\w+)*",
        "{date_pattern}": r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})|(\d{1,2}[-/]\d{1,2}[-/]\d{4})",
        "{time_pattern}": r"(?:[01]\d|2[0-3]):[0-5]\d",
        "{ipv4_pattern}": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "{url_pattern}": r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
        "{ssn_pattern}": r"\d{3}-\d{2}-\d{4}",
        "{license_plate_pattern}": r"[A-Z0-9]{2,}-[A-Z0-9]{2,}",
        "{zip_code_pattern}": r"\d{5}(-\d{4})?",
        "{vin_pattern}": r"[A-HJ-NPR-Z0-9]{17}",
        "{iban_pattern}": r"[A-Z]{2}\d{2}[A-Z0-9]{1,30}",
        "{driver_license_pattern}": r"[A-Z]{1,2}-\d{4,9}"
    }



    def __init__(self, completion_model:OpenAIChatModel):
        
        self.nlp = spacy.load("en_core_web_sm")
        self.ner_prompt = load_file("files/ner.txt")
        self.completion_model = completion_model

    def regex_anonymization(self, text: str) -> str:

        for replacement, pattern in self.REGEX_PATTERN.items():
            text = re.sub(pattern, replacement, text)
        
        return text

    def ner_anonymization(self, text: str) -> str:
        doc = self.nlp(text)
        for entity in doc.ents:
            placeholder = self.NER_PLACEHOLDER.get(entity.label_)
            if placeholder:
                text = text.replace(entity.text, placeholder)
        return text

    def llm_anonymization(self, text: str) -> str:

        completion = self.completion_model.ask(
            system_prompt=self.ner_prompt,
            user_prompt=text,
            max_tokens=126,
            temperature=0
        ) 

        return completion

    def anonymize(self, text: str) -> str:

        text = self.regex_anonymization(text)
        text = self.ner_anonymization(text)
        return self.llm_anonymization(text)



import os
from pydantic import BaseModel
from typing import Optional, List, Any, Dict, Optional
from enum import Enum
from pandas import DataFrame

### Schema types ###


class Status(str, Enum):
    """Status of a resource"""
    predefined = "predefined"
    new = "new"
    seen = "seen"
    accepted = "accepted"
    archived = "archived"


class ResourceType(str, Enum):
    verse = "verse"
    word = "word"
    phrase = "phrase"
    sentence = "sentence"
    lexical_entry = "lexical_entry"
    note = "note"
    # TODO: not sure what exactly we will need here. OpenText entity, speech act, definition, etc.?


class Resource(BaseModel):
    """Identified resources are meant to live in a PostGres resource"""
    id: str
    type: ResourceType
    language_code: str # Top-level because most resources are language-specific
    content: str
    status: Status
    metadata: dict[str, Any]
    
    def format(self) -> str:
        """Necessary function for formatting the resource content plus any metadata that should be included but can be accessed apart from the formatted string"""
        pass
    
    
class Prompt(BaseModel):
    action: str
    inputs: list[Resource]
    # context: Dict[str, Any] # ? needed?
    goal: str
    method: str
    # template: list[str] # [goal, inputs, method, action] # TODO: if we don't use another library
    template: str # TODO: if we use something like guidance or outlines


class Translation(BaseModel):
    """
    A source, target pair for translation.
    """
    id: str
    source: Resource # starting place for the translation instance
    target: Resource # output of the translation instance
    bridge_translations: list[Resource]
    prompt: Prompt
        
    def revise_target(self, approach: str):
        pass
    
    def translate_from_source(self):
        pass
    

class ResourceCollection(BaseModel):
    id: str
    metadata: dict[str, Any]
    resources: list[Resource]
    

### Deprecated types ###

class TranslationTriplet(BaseModel):
    """
    TranslationTriplet is a single translation triplet
    
    "source": "οὕτως γὰρ ἐντέταλται ἡμῖν ὁ Κύριος Τέθεικά σε εἰς φῶς ἐθνῶν τοῦ εἶναί σε εἰς σωτηρίαν ἕως ἐσχάτου τῆς γῆς.",
    "bridge_translation": ["For this is what the Lord has commanded us: ‘I have made you a light for the Gentiles, to bring salvation to the ends of the earth.’”",]
    "target": "Anayabin Regah ana obaiyunen tur biti iti na’atube eo, ‘Ayu kwa ayasairi Ufun Sabuw hai marakaw isan, saise kwa i boro yawas kwanab kwanatit kwanan tafaram yomanin kwanatit.’"
    """
    source: str
    bridge_translation: str
    target: str
    
    def to_dict(self):
        return {
            "source": self.source,
            "bridge_translation": self.bridge_translation,
            "target": self.target,
            }
 

class VerseMap(BaseModel): # FIXME: instance
    # An object, where each key is a vref string, and each value is a Verse type
    verses: dict[str, TranslationTriplet]

    
### API response types ###

class Message(BaseModel):
    """Message includes content and role (typically system|user|assistant)"""
    role: str
    content: str


class RequestModel(BaseModel):
    messages: List[Message]

    
class Choice(BaseModel):
    """Choice is a single choice from a chat response, in case multiple choices are requested"""
    index: int
    message: Message
    finish_reason: str


class ChatResponse(BaseModel):
    """ChatResponse is the response from OpenAI's v1/chat/completions API"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Dict[str, int]


class InputReceived(BaseModel):
    VerseMap


class AIResponse(BaseModel):
    """AIResponse is a response from OpenAI's v1/chat/completions API plus inputs and relevant verse reference"""
    input_received: InputReceived
    hypothesis_vref: str
    prediction: ChatResponse

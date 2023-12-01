from langchain.prompts import PromptTemplate,FewShotPromptTemplate
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector.base import BaseExampleSelector
from typing import Dict, List

bio_summary_examples = [
    {"person": "Napoleon", "bio": "Napoleon Bonaparte[a] (born Napoleone Buonaparte; 15 August 1769 – 5 May 1821), later known by his regnal name Napoleon I,[b] was a Corsican-born French military commander and political leader who rose to prominence during the French Revolution and led successful campaigns during the Revolutionary Wars. He was the de facto leader of the French Republic as First Consul from 1799 to 1804, then Emperor of the French from 1804 until 1814 and again in 1815. Napoleon's political and cultural legacy endures to this day, as a highly celebrated and controversial leader. He initiated many liberal reforms that have persisted in society, and is considered one of the greatest military commanders in history. His campaigns are still studied at military academies worldwide. Between three and six million civilians and soldiers died in what became known as the Napoleonic Wars.[2][3]"},
    {"person": "Alexander the Great", "bio": "Alexander III of Macedon (Ancient Greek: Ἀλέξανδρος, romanized: Alexandros; 20/21 July 356 BC – 10/11 June 323 BC), commonly known as Alexander the Great,[a] was a king of the ancient Greek kingdom of Macedon.[a] He succeeded his father Philip II to the throne in 336 BC at the age of 20, and spent most of his ruling years conducting a lengthy military campaign throughout Western Asia and Egypt. By the age of 30, he had created one of the largest empires in history, stretching from Greece to northwestern India.[2] He was undefeated in battle and is widely considered to be one of history's greatest and most successful military commanders.[3][4]"},
    {"person": "Julius Caesar", "bio": "Gaius Julius Caesar (/ˈsiːzər/; Latin: [ˈɡaːiʊs ˈjuːliʊs ˈkae̯sar]; 12 July 100 BC – 15 March 44 BC) was a Roman general and statesman. A member of the First Triumvirate, Caesar led the Roman armies in the Gallic Wars before defeating his political rival Pompey in a civil war, and subsequently became dictator from 49 BC until his assassination in 44 BC. He played a critical role in the events that led to the demise of the Roman Republic and the rise of the Roman Empire."}
]
bio_summary_template ="""
Person: {person}
Bio: {bio}
"""

class StartsWithASelector(BaseExampleSelector)

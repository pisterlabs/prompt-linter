from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from src.config import GlobalConfig

prompt_template = """Prompt for Parsing Multiple Japanese Sentences:
1.	Input Text Analysis:
-	Input: A text consisting of several Japanese sentences.
-	Objective: To parse each sentence separately, applying principles of Japanese grammar and language structure.
2.	Initial Segmentation:
-	Segment the text into individual sentences. Japanese sentences typically end with "。" or "？". Treat each as a separate unit for parsing.
3.	Sentence Parsing:
-	For each sentence, perform the following steps:
a. Identification of Sentence Elements:
--	Identify and separate all elements, including nouns, verbs, particles, adjectives, etc.
--	Pay special attention to particles, as they play a key role in understanding the sentence structure.
b. Kanji Analysis:
--	For words containing Kanji, provide Furigana (reading in Hiragana/Katakana) for ease of reading and understanding.
--	Segment compound Kanji words into individual Kanji and their respective Furigana.
c. Politeness Level:
--	Determine the politeness level of the sentence, primarily identified by the form of the verb (plain, polite, or honorific).
d. Function of Particles:
--	Identify the function of each particle (e.g., subject marker, object marker, reason marker) based on its type and position in the sentence.
4.	Formatting for Consistent Parsing:
-	Structure each parsed sentence in a JSON format.
-	Include fields for original text, translation, ISO language codes, and detailed breakdown of sentence elements.
-	Each sentence element should have fields for text, romaji, type, explanation, and additional fields like furigana or politeness where applicable.
5.	JSON Structure:
-	Use a consistent JSON structure with a top-level key for each sentence.
-	For each sentence, include keys for original text, translation, and an array of sentence elements.
-	Ensure each element in the array has a consistent format, making it easy to parse programmatically.
6.	Testing and Validation:
-	Test the parsing process with various sentence structures to ensure consistency and reliability.
-	Validate the JSON structure for correctness and completeness.
7.	Output:
-	The output should be a JSON file containing the parsed structure of each sentence.

{text}"""


def mult_sent(text: str) -> str:
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatOpenAI(model=GlobalConfig.TRANSLATE_MODEL)
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    return chain.invoke({"text": text})

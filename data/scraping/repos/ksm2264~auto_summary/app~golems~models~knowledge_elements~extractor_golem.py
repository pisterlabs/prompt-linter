from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from app.golems.models.knowledge_elements.templates import extractor_template
from app.golems.models.knowledge_elements.parser import parse_elements
from app.sample_assets.papers import sample_text_chunks

llm = ChatOpenAI(model_name = 'gpt-4')
golem_chain = LLMChain(llm = llm, prompt = extractor_template)

class ExtractorGolem:
    def __init__(self):
        self.chain = golem_chain
 
    def get_element_list(self, text_chunk: str) -> List[str]:

        raw_output = golem_chain.predict(
            text_chunk = text_chunk
        )

        return parse_elements(raw_output)
    
if __name__ == '__main__':
    
    golem = ExtractorGolem()

    for text in sample_text_chunks:

        elements = golem.get_element_list(text)

        print(f'Element list: {elements}')
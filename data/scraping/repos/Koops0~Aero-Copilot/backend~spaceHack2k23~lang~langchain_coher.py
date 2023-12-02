from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.embeddings import CohereEmbeddings
from spaceHack2k23.settings import *

# paragraph = '''Glass and other brittle materials typically have large scatter in material properties. Additionally,
#     brittle materials are prone to sudden failure without prior warning. These features require a
#     higher safety factor than standard, well-characterized materials.
#     a. Annealed glass shall be designed to a minimum ultimate factor of safety of 3.0
#     applied to the maximum combined limit load (section 4.3.9 of this Standard) at the beginning of
#     life.
#     b. Annealed glass shall be designed to a minimum ultimate factor of safety of 1.4
#     applied to the maximum combined limit load (section 4.3.9 of this Standard) at the end of life.'''


def getTechnicalInformation(paragraph):
    template = """You are a experienced scientist reading a document.  
    You are tasked with identifying and extracting all the technical keywords, phrases and concepts. 
    Be precise with your answer. 
    Your output should be a machine readable JSON list with a list of all the technical keywords, phrases and concepts with this format:
    {{
        "technical keywords": [

        ],
        "technical phrases": [

        ],
        "technical concepts": [

        ]
    }}
    Do not include any text other than a Python JSON list.

    Given the information above, extract all the technical keywords, phrases and concepts from the below text:

    {paragraph}"""

    prompt = PromptTemplate(template=template, input_variables=["paragraph"])
    llm = Cohere(cohere_api_key=COHERE_KEY)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    output = llm_chain.run(paragraph)
    return output

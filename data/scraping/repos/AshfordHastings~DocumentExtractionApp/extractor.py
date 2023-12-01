from langchain.llms import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.schema.output_parser import BaseOutputParser, StrOutputParser

from langchain_app.contexts import ExtractionContext, DocumentExtractionContext, StringExtractionContext
from langchain_app.templates import SimpleExtractionPromptTemplate, ExtractionPromptTemplate

class ExtractionChain:
    def __init__(self, context:ExtractionContext, template:ExtractionPromptTemplate, model:BaseLLM, parser:BaseOutputParser):
        self.context = context
        self.template = template
        self.model = model
        self.parser = parser

    def extract(self, data):
        return self.generate_chain().invoke(data)

    def generate_chain(self):
        return (
            {"context": self.context.get_runnable()}
            | self.template.get_runnable()
            | self.model
            | self.parser
        )
    
class SimpleDocumentStringExtractionChainBuilder:
    def __init__(self, source_data, schema):
        self.source_data = source_data
        self.schema = schema
        self.context = StringExtractionContext(source_data)
        self.template = SimpleExtractionPromptTemplate(schema)
        self.model = OpenAI()
        self.parser = StrOutputParser()

    def build(self):
        return ExtractionChain(self.context, self.template, self.model, self.parser)
    
class SimpleDocumentPDFExtractionChainBuilder:
    def __init__(self, artifact, schema):
        self.source_data = artifact
        self.schema = schema
        self.context = DocumentExtractionContext(artifact)
        self.template = SimpleExtractionPromptTemplate(schema)
        self.model = OpenAI()
        self.parser = StrOutputParser()

    def build(self):
        return ExtractionChain(self.context, self.template, self.model, self.parser)

def extract_data_inline(data, schema):
    extraction_chain = SimpleDocumentStringExtractionChainBuilder(data, schema).build()
    return extraction_chain.extract(data="")



def extract_data_pdf(artifact, schema):
    extraction_chain = SimpleDocumentPDFExtractionChainBuilder(artifact, schema).build()
    return extraction_chain.extract(data="")

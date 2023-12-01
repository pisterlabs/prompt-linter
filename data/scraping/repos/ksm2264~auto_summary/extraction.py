from langchain.prompts import PromptTemplate

extraction_template = PromptTemplate(
    template='''
    you extract key concepts from chunks of a research paper.
    
    This is the past few chunks:
    {history}

    This is the next chunk
    {chunk}

    ''',
    input_variables=['history', 'chunk']
)
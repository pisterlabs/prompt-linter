import sys

from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

ollama = Ollama(base_url="http://localhost:11434", model="llama2")

pt_chapter_name = PromptTemplate(
    input_variables=["theme"],
    template="""
    You are a huge fan of Warhammer 40,000. You know everything about the setting. 
    I need a good name for a new Space Marine chapter with the following
    theme {theme}. When you answer only reply with the new chapter's name. Do
    not include any extra text, jokes, commentary, preambles, etc. Just reply with the
    name, by itself.
    """,
)

chapter_name_chain = LLMChain(
    llm=ollama, prompt=pt_chapter_name, output_key="chapter_name"
)

pt_space_marine_names = PromptTemplate(
    input_variables=["chapter_name"],
    template="""
    You are a huge fan of Warhammer 40,000. You know everything about the setting. 
    Make up some Warhammer 40,000 space marine names that would fit in a chapter 
    named {chapter_name}
    """,
)

space_marines_names = LLMChain(llm=ollama, prompt=pt_space_marine_names)

chain = SimpleSequentialChain(
    chains=[chapter_name_chain, space_marines_names], verbose=True
)
response = chain.run(sys.argv[1])

print(response)

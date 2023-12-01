from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import SimpleSequentialChain
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

with open("result.txt") as f:
    long_text = f.read()
max_tokens: int = 60
llm = OpenAI(temperature=0)
text_splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=20)

summary_chain = load_summarize_chain(llm, chain_type="map_reduce")

promptSubject = PromptTemplate(
    input_variables=["text"],
    template="""\"\"\"{text}\"\"\"\
上記の内容を要約してください：\n\n* """,
)
chainSubject = LLMChain(llm=llm, prompt=promptSubject)

overall_chain_map_reduce = SimpleSequentialChain(chains=[summary_chain, chainSubject])
subject = overall_chain_map_reduce.run(text_splitter.create_documents([long_text]))
print(subject)

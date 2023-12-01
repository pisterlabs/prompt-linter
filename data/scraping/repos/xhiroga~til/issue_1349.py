from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI


load_dotenv()

urls = [
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-8-2023",
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-9-2023",
]

loader = UnstructuredURLLoader(urls=urls)

data = loader.load()

llm = OpenAI(temperature=0)
chain = load_summarize_chain(llm, chain_type="map_reduce")
print(chain.run(data))

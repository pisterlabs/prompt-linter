from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate

base_embeddings = OpenAIEmbeddings()
llm = OpenAI()

# Load with `web_search` prompt
embeddings = HypotheticalDocumentEmbedder.from_llm(llm, base_embeddings, "web_search")

# Now we can use it as any embedding class!
result = embeddings.embed_query("Where is the Taj Mahal?")

multi_llm = OpenAI(n=4, best_of=4)

embeddings = HypotheticalDocumentEmbedder.from_llm(
    multi_llm, base_embeddings, "web_search"
)

result = embeddings.embed_query("Where is the Taj Mahal?")

prompt_template = """Please answer the user's question about the most recent state of the union address
Question: {question}
Answer:"""
prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
llm_chain = LLMChain(llm=llm, prompt=prompt)

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

docsearch = Chroma.from_texts(texts, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)

print(docs[0].page_content)
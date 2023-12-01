from langchain.document_loaders.csv_loader import CSVLoader
from rich import print

csv_file = "./results/filtered_readme_repos_20230329-112511.csv"
loader = CSVLoader(file_path=csv_file)
data = loader.load()

from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI

index = VectorstoreIndexCreator().from_loaders([loader])
query = "how many repo urls are there?"
print(index.query(query))

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

agent = create_csv_agent(OpenAI(temperature=0), csv_file, verbose=True)
agent.run(query)
print(agent.run("how many rows are there?"))

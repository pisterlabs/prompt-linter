import openai
import yaml
from langchain.document_loaders import DirectoryLoader

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

loader = DirectoryLoader(config['path'], glob = "**/*.pdf")                                                   
docs = loader.load()                                                                         
print(f"Document Count: {len(docs)}")
                                                                                                                                                                        
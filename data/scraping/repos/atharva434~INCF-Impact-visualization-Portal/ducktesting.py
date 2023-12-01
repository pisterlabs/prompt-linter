from langchain.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
print(search.run("The exact number of people suffering from stroke in the world"))
from langchain.utilities import SerpAPIWrapper

search = SerpAPIWrapper()

result = search.run("Obama's first name?")

print('result=', result)


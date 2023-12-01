from langchain.document_loaders import SeleniumURLLoader

class Data_Collector:
  def Collector(self,link):
    input_string = link
    Url_list = input_string.split(" ")
    urls = Url_list
    loader = SeleniumURLLoader(urls = urls)
    data = loader.load()
    print("The data you requested is as follows : ")
    print(data)
    return data

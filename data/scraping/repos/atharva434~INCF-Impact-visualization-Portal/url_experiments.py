from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import cohere
from langchain.llms import Cohere
import nltk
import json

#give this model internet access for getting real time data
llm = Cohere(cohere_api_key="4aJ9yWbIrOzI2W5LZeLeIdin2AYMpkq18PffLuvi",temperature=0)
# import nltk
# nltk.download('averaged_perceptron_tagger')
urls = [
    "https://www.thevirtualbrain.org/tvb/zwei"
]

loader = UnstructuredURLLoader(urls=urls) # try other loaders

docs = loader.load()
print()
tex=str(docs[0])


def diseases(tex):
    
    prompt=f"write the mental disorders mentioned in {tex} if no mental disorders present say 'no disease found'"
    diseases=llm(prompt)
    # nltk_tokens = nltk.word_tokenize(diseases)
    return diseases
    # print(nltk_tokens)
def research_params(tex):
    prompt=f"""Give me a structured output in json format covering the aim,use of project, and the real world impact of this project and expand in detail on the impact {tex} 
 the project description is {tex}"""
    params=llm(prompt)
    params=json.loads(params)
    print(params["impact"])
    return params
def impact_count(disease):
    prompt2=f"In{disease} there is list of diseases Find the number of people suffering from each respective disease in millions and store it in json format with key as name of disease no need to give background information"
    return llm(prompt2)

# prompt_check=f"write 'yes' if any mental disorder is detected in {tex} otherwise return 'no'"
# print(llm(prompt_check))
list_diseases=diseases(tex)
print(list_diseases)
if "no"not in list_diseases:
    print(impact_count(list_diseases))

# else:
#     print(research_params(tex))

    # print("sorry")


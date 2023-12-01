from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
import requests

def get_wiki_data(title, first_paragraph_only):
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles={title}"
    if first_paragraph_only:
        url += "&exintro=1"
    data = requests.get(url).json()
    return Document(
        page_content=list(data["query"]["pages"].values())[0]["extract"],
        metadata={"source": f"https://en.wikipedia.org/wiki/{title}"},
    )

sources = [
    get_wiki_data("Unix", True),
    get_wiki_data("Microsoft_Windows", True),
    get_wiki_data("Linux", True),
    get_wiki_data("Seinfeld", True),
]

chain = load_qa_with_sources_chain(OpenAI(temperature=0))

def print_answer(question):
    print("[AI]: " +
        chain(
            {
                "input_documents": sources,
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )


while True:
    user_input = input("[Human ('e' to exit)]: ")
    if user_input == "e":
        print("[AI]: kthxbai")
        break
    print_answer(user_input)
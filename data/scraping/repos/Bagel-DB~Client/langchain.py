from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from langchain.vectorstores import Bagel
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from termcolor import colored


def fetch_content(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        content = page.content()
        browser.close()
    return BeautifulSoup(content, "html.parser").get_text()


if __name__ == "__main__":
    url = input("Enter the URL: ")
    text = fetch_content(url)
    cluster = Bagel.from_texts(cluster_name="testing_langchain", texts=[text])

    qa_chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=cluster.as_retriever())

    while True:
        input_query = input("your query: ")
        if input_query == "q":
            break
        print(colored(qa.run(input_query), "red"))
        print("-" * 40)

    cluster._client.delete_cluster("testing_langchain")
    print("Done!")

import os
from os.path import join
from notion_client import Client
import arxiv
import questionary
import textwrap
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory, FileHistory

import requests
from langchain.document_loaders import PDFMinerLoader, PyPDFLoader, BSHTMLLoader, UnstructuredURLLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings  # for creating embeddings
from langchain.vectorstores import Chroma  # for the vectorization part
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from notion_tools import QA_notion_blocks, clean_metadata, print_entries, save_qa_history, load_qa_history, print_qa_result

history = FileHistory("notion_arxiv_history.txt")
session = PromptSession(history=history)

chathistory = FileHistory("qa_chat_history.txt")
chatsession = PromptSession(history=chathistory)

database_id = "d3e3be7fc96a45de8e7d3a78298f9ccd"
notion = Client(auth=os.environ["NOTION_TOKEN"])

PDF_DOWNLOAD_ROOT = r"E:\OneDrive - Harvard University\openai-emb-database\arxiv_pdf"
EMBED_ROOTDIR = r"E:\OneDrive - Harvard University\openai-emb-database\Embed_data"


def arxiv_entry2page_blocks(paper: arxiv.arxiv.Result):
    title = paper.title
    authors = [author.name for author in paper.authors]
    pubyear = paper.published
    abstract = paper.summary
    arxiv_id = paper.entry_id.split("/")[-1]
    abs_url = paper.entry_id
    page_prop = {
        'Name': {
            "title": [
                {
                    "text": {
                        "content": f"[{arxiv_id}] {title}"
                    }
                }],
        },
        "Author": {
            "multi_select": [
                {'name': name} for name in authors
            ]
        },
        'Publishing/Release Date': {
            'date': {'start': pubyear.date().isoformat(), }
        },
        'Link': {
            'url': abs_url
        }
    }
    content_block = [{'quote': {"rich_text": [{"text": {"content": abstract}}]}},
                     {'heading_2': {"rich_text": [{"text": {"content": "Related Work"}}]}},
                     {'paragraph': {"rich_text": [{"text": {"content": ""}}]}},
                     {'heading_2': {"rich_text": [{"text": {"content": "Techniques"}}]}},
                     {'paragraph': {"rich_text": [{"text": {"content": ""}}]}},
                     ]
    return page_prop, content_block


def arxiv_entry2page(database_id, paper: arxiv.arxiv.Result):
    page_prop, content_block = arxiv_entry2page_blocks(paper)
    new_page = notion.pages.create(parent={"database_id": database_id}, properties=page_prop)
    notion.blocks.children.append(new_page["id"], children=content_block)
    return new_page["id"], new_page


def print_arxiv_entry(paper: arxiv.arxiv.Result):
    title = paper.title
    authors = [author.name for author in paper.authors]
    pubyear = paper.published
    abstract = paper.summary
    arxiv_id = paper.entry_id.split("/")[-1]
    abs_url = paper.entry_id
    print(f"[{arxiv_id}] {title}")
    print("Authors:", ", ".join(authors))
    print("Published:", pubyear.date().isoformat())
    print("Abstract:")
    print(textwrap.fill(abstract, width=100))
    print("comments:", paper.comment)
    print("URL:", abs_url)


def blocks2text(blocks):
    if "results" in blocks:
        blocks = blocks["results"]
    for block in blocks:
        if block["type"] == "paragraph":
            for parts in block["paragraph"]["rich_text"]:
                print(textwrap.fill(parts["plain_text"], width=100))

        elif block["type"] == "heading_2":
            for parts in block["heading_2"]["rich_text"]:
                print(textwrap.fill(parts["plain_text"], width=100))

        elif block["type"] == "quote":
            for parts in block["quote"]["rich_text"]:
                print(textwrap.fill(parts["plain_text"], width=100))
        else:
            print(block["type"])


def fetch_K_results(search_obj, K=10, offset=0):
    """Fetches K results from the search object, starting from offset, and returns a list of results."""
    results = []
    try:
        for entry in search_obj.results(offset=offset):
            results.append(entry)
            if len(results) >= K:
                break
    except StopIteration:
        pass
    return results


def add_to_notion(paper: arxiv.arxiv.Result):
    title = paper.title
    arxiv_id = paper.entry_id.split("/")[-1]
    # check if entry already exists in Notion database
    results_notion = notion.databases.query(database_id=database_id,
                                            filter={"property": "Link", "url": {"contains": arxiv_id}})
    if len(results_notion["results"]) == 0:
        # page does not exist, create a new page
        print(f"Adding entry paper {arxiv_id}: {title}")
        page_id, page = arxiv_entry2page(database_id, paper)
        print(f"Added entry {page_id} for arxiv paper {arxiv_id}: {title}")
        print_entries([page], print_prop=("url",))
        return page_id, page
    else:
        # page already exists, ask user if they want to update the page
        print_entries(results_notion, print_prop=("url",))
        print("Entry already exists as above. ")
        if len(results_notion["results"]) == 1:
            page_id, page = results_notion["results"][0]["id"], results_notion["results"][0]
            #TODO: update page with entry
            return page_id, page
        else:
            page_id = questionary.select("Select paper:",
                         choices=[page["id"] for page in results_notion["results"]]).ask()
            page = [page for page in results_notion["results"] if page["id"] == page_id][0]
            #TODO: update page with entry
            return page_id, page


def arxiv_paper_download(arxiv_id, pdf_download_root=""):
    """Downloads the arxiv paper with the given arxiv_id, and returns the path to the downloaded pdf file."""
    ar5iv_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    # try getting ar5iv page first
    r = requests.get(ar5iv_url, allow_redirects=True, )
    if r.url.startswith("https://ar5iv.labs.arxiv.org/html"):
        # if not redirected, then ar5iv page exists
        # then download html to parse
        print(f"Downloading {r.url}...")
        open(join(pdf_download_root, f"{arxiv_id}.html"), 'wb').write(r.content)
        print("Saved to", join(pdf_download_root, f"{arxiv_id}.html"))
        loader = BSHTMLLoader(join(pdf_download_root, f"{arxiv_id}.html"),
                              open_encoding="utf8", bs_kwargs={"features": "html.parser"})
        pages = loader.load_and_split()
    else:
        # if redirected, then ar5iv page does not exist
        # save pdf instead
        print(f"redirected to {r.url}")
        print("ar5iv not found, downloading pdf instead ")
        r = requests.get(pdf_url, allow_redirects=True, )
        open(join(pdf_download_root, f"{arxiv_id}.pdf"), 'wb').write(r.content)
        print("Saved to", join(pdf_download_root, f"{arxiv_id}.pdf"))
        loader = PyPDFLoader(join(pdf_download_root, f"{arxiv_id}.pdf"))
        # loader = PDFMinerLoader(pdf_path)
        pages = loader.load_and_split()
    return pages


def notion_paper_chat(arxiv_id, pages=None, save_page_id=None, embed_rootdir=""):
    if save_page_id is None:
        print("No page id provided, no chat history will be saved to Notion.")

    if pages is None:
        print("No pages provided, downloading paper from arxiv...")
        pages = arxiv_paper_download(arxiv_id)

    # create embedding directory
    embed_persist_dir = join(embed_rootdir, arxiv_id)
    qa_path = embed_persist_dir + "_qa_history"
    os.makedirs(qa_path, exist_ok=True)
    # create embeddings
    embeddings = OpenAIEmbeddings()
    if os.path.exists(embed_persist_dir):
        print("Loading embeddings from", embed_persist_dir)
        vectordb = Chroma(persist_directory=embed_persist_dir, embedding_function=embeddings)
        print("Loading Q&A history from", qa_path)
        chat_history, queries, results = load_qa_history(qa_path)
        while True:
            question = questionary.select("Select Q&A history:", choices=["New query"] + queries,
                                          default="New query").ask()
            if question == "New query":
                break
            else:
                print("Q:", question)
                result = results[queries.index(question)]
                print_qa_result(result, )
    else:
        print("Creating embeddings and saving to", embed_persist_dir)
        vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                         persist_directory=embed_persist_dir, )
        vectordb.persist()

    chat_temperature = questionary.text("Sampling temperature for ChatGPT?", default="0.3").ask()
    chat_temperature = float(chat_temperature)
    # ref_maxlen = questionary.text("Max length of reference document?", default="300").ask()
    ref_maxlen = 200
    pdf_qa_new = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=chat_temperature, model_name="gpt-3.5-turbo"),
        vectordb.as_retriever(), return_source_documents=True, max_tokens_limit=4000)

    # Q&A loop with ChatOpenAI
    with get_openai_callback() as cb:
        while True:
            try:
                # query = "For robotics purpose, which algorithm did they used, PPO, Q-learning, etc.?"
                query = chatsession.prompt("Question: ", multiline=False)
                # query = questionary.text("Question: ", multiline=True).ask()
                if query == "" or query is None:
                    if questionary.confirm("Exit?").ask():
                        break
                    else:
                        continue

                result = pdf_qa_new({"question": query, "chat_history": ""})

                print_qa_result(result)
                # local save qa history
                save_qa_history(query, result, qa_path)
                # save to notion
                if save_page_id is not None:
                    answer = result["answer"]
                    refdocs = result['source_documents']
                    refstrs = [str(refdoc.metadata) + refdoc.page_content[:ref_maxlen] for refdoc in refdocs]
                    try:
                        notion.blocks.children.append(save_page_id, children=QA_notion_blocks(query, answer, refstrs))
                    except Exception as e:
                        print("Failed to save to notion")
                        print(e)
                        refstrs_meta = [str(refdoc.metadata) for refdoc in refdocs]
                        notion.blocks.children.append(save_page_id, children=QA_notion_blocks(query, answer, refstrs_meta))
            except KeyboardInterrupt:
                break
        # End of chat loop
        print(f"Finish conversation")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
# query = "2106.05963"
# query = "au:Yann LeCun"
# Logic:
# Ctrl-C in the navigation loop to exit and start a new query
# Ctrl-C in the query prompt to exit the program
# Up/Down to navigate through prompts and query history
MAX_RESULTS = 35
while True:
    try:
        cnt = 0
        query = session.prompt("Enter arXiv ID or query str: ", multiline=False)
        search_obj = arxiv.Search(query, )
        results_arxiv = fetch_K_results(search_obj, K=MAX_RESULTS, offset=cnt)
        if len(results_arxiv) == 0:
            print("No results found.")
            continue
        elif len(results_arxiv) == 1:
            paper = results_arxiv[0]
            arxiv_id = paper.entry_id.split("/")[-1]
            print_arxiv_entry(paper)
            # Add the entry if confirmed
            if questionary.confirm("Add this entry?").ask():
                page_id, _ = add_to_notion(paper)
                # if questionary.confirm("Save arxiv pdf?").ask():
                if questionary.confirm("Q&A Chatting with this file?").ask():
                    pages = arxiv_paper_download(arxiv_id, pdf_download_root=PDF_DOWNLOAD_ROOT)
                    notion_paper_chat(arxiv_id, pages=pages, save_page_id=page_id, embed_rootdir=EMBED_ROOTDIR)

        elif len(results_arxiv) > 1:
            # multiple results found, complex logic to navigate through results
            last_selection = None  # last selected result to highlight
            while True:
                # looping of results and pages, navigating through search results
                print("Multiple results found. Please select one:")
                choices = [f"{i + 1}: [{paper.entry_id.split('/')[-1]}] {paper.title} " for i, paper in enumerate(results_arxiv)]
                if len(results_arxiv) == MAX_RESULTS:
                    choices.append("0: Next page")
                if cnt > 0:
                    choices.append("-1: Prev page")
                selection = questionary.select("Select paper:", choices=choices, default=None if last_selection is None
                                               else choices[last_selection]).ask()
                selection = int(selection.split(":")[0])
                if selection == 0:
                    cnt += MAX_RESULTS
                    results_arxiv = fetch_K_results(search_obj, K=MAX_RESULTS, offset=cnt)
                    continue
                if selection == -1:
                    cnt -= MAX_RESULTS
                    results_arxiv = fetch_K_results(search_obj, K=MAX_RESULTS, offset=cnt)
                    continue
                else:
                    paper = results_arxiv[int(selection) - 1]
                    last_selection = int(selection) - 1
                    arxiv_id = paper.entry_id.split("/")[-1]
                    print_arxiv_entry(paper)
                    if questionary.confirm("Add this entry?").ask():
                        # Add the entry if confirmed
                        page_id, _ = add_to_notion(paper)
                        # if questionary.confirm("Save arxiv pdf?").ask():
                        if questionary.confirm("Q&A Chatting with this file?").ask():
                            pages = arxiv_paper_download(arxiv_id, pdf_download_root=PDF_DOWNLOAD_ROOT)
                            notion_paper_chat(arxiv_id=arxiv_id, pages=pages, save_page_id=page_id,
                                              embed_rootdir=EMBED_ROOTDIR)
                    # if questionary.confirm("Back to the list").ask():
                    #     continue
                    # else:
                    #     break

    except Exception as e:
        print(e)
        continue

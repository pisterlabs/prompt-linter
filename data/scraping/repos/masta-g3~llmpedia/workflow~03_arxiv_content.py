import sys, os
from dotenv import load_dotenv
from langchain.document_loaders import ArxivLoader
from tqdm import tqdm

load_dotenv()
sys.path.append(os.environ.get("PROJECT_PATH"))

import utils.paper_utils as pu
import utils.db as db

db_params = pu.db_params
data_path = os.path.join(os.environ.get("PROJECT_PATH"), "data")


def main():
    """Load summaries and add missing ones."""
    arxiv_code_map = db.get_arxiv_title_dict()
    done_path = os.path.join(data_path, "arxiv_text")
    done_codes = [f.replace(".txt", "") for f in os.listdir(done_path)]
    pending_codes = list(set(arxiv_code_map.keys()) - set(done_codes))
    pending_codes_map = {k: v for k, v in arxiv_code_map.items() if k in pending_codes}

    items_added = 0
    for k, v in tqdm(pending_codes_map.items()):
        doc = pu.search_arxiv_doc(k)
        if doc is None:
            doc = pu.search_arxiv_doc(v)
        if doc is None:
            print(f"Axiv code {v} not found.")
            continue
        doc_content = doc.page_content
        pu.store_local(doc_content, k, "arxiv_text", True, "txt")
        # print(f"\nFile {k} saved.")
        items_added += 1

    print(f"Process complete. Added {items_added} items in total.")


if __name__ == "__main__":
    main()

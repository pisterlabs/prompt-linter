from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from bs4 import BeautifulSoup
import tiktoken, markdown, re

# calculate token size
ENCODING = tiktoken.encoding_for_model('gpt-3.5-turbo')

headers_to_split_on = [
    ("##", 2)
]

# MD splits
MD_SPLITTER = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Char-level splits
CHUNK_SIZE = 1024 #250
CHUNK_OVERLAP = 0 #30
TEXT_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=['\n\n', '\n'],
    model_name='gpt-3.5-turbo'
)

# Split
def markdown_layer_split(text, layer=1):
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('#'*layer, layer)])
    md_header_splits = markdown_splitter.split_text(text)
    return md_header_splits

def split_docs(docs, max_length=CHUNK_SIZE):
    splits = []
    for i, doc in enumerate(docs):
        if len(ENCODING.encode(doc.page_content)) < CHUNK_SIZE:
            splits.append(doc)
        else:
            text, metadata = doc.page_content, doc.metadata

            layer = max([l for l in metadata.keys()]) + 1 if len(metadata.keys()) > 0 else 3

            # split on markdown headers
            md_header_splits = markdown_layer_split(text, layer)

            # if no headers, split on next layer
            if len(md_header_splits) == 1:
                i = 1
                while len(ENCODING.encode(md_header_splits[0].page_content)) > CHUNK_SIZE:
                    md_header_splits = markdown_layer_split(text, layer + i)
                    i += 1
                    if i > 2:
                        md_header_splits = [Document(page_content=d) for d in TEXT_SPLITTER.split_text(text)]
                        break

            for chunk in md_header_splits:
                content = chunk.page_content
                meta2 = chunk.metadata.copy()
                meta2.update(metadata.copy())
                new_doc = Document(page_content=content, metadata=meta2)
                if len(ENCODING.encode(content)) < CHUNK_SIZE:
                    splits.append(new_doc)
                else:
                    splits.extend(split_docs([new_doc], ENCODING))
    return splits

# combine pages whose token size is less than CHUNK_SIZE(1024)
def combine_pages(contents, metas, header=""):
    ans3 = []
    idx = 1
    page, meta = contents[0], metas[0].copy()
    k_v = [i for i in meta.items()]
    k_v.sort(reverse=True)
    for k, v in k_v:
        page = f"{k * '#'} {v}\n" + page

    while idx < len(metas):
        if len(ENCODING.encode(contents[idx] + page)) < CHUNK_SIZE:
            page += '\n'
            # update meta
            k_v = [i for i in metas[idx].items()]
            k_v.sort()
            for k, v in k_v:
                if k in meta and v not in meta[k]:
                    meta[k] += f" | {v}"
                    page += f"{k * '#'} {v}\n"
                elif k not in meta:
                    meta[k] = v
                    page += f"{k * '#'} {v}\n"
            page += contents[idx]
            # page += '\n\n' + contents[idx]
            # for k, v in metas[idx].items():
            #     if k in meta and v not in meta[k]:
            #         meta[k] += f" | {v}"
            #     else:
            #         meta[k] = v
        else:
            ans3.append(Document(page_content=page, metadata=meta.copy()))
            page, meta = contents[idx], metas[idx].copy()
            k_v = [i for i in meta.items()]
            k_v.sort(reverse=True)
            for k, v in k_v:
                page = f"{k * '#'} {v}\n" + page
        idx += 1

    if page:
        ans3.append(Document(page_content=page, metadata=meta.copy()))

    for a in ans3:
        if len(a.metadata.keys()) == 0:
            a.metadata[1] = header

    return ans3

def md_to_str(md, head: str):
    soup = BeautifulSoup(markdown.markdown(md), features="html.parser")

    for a in soup.find_all('a', href=True):
        a.decompose()

    text = soup.get_text().replace('[TOC]', '').strip()
    for h in head.split(' | '):
        text = text.replace(h, '')
    return re.sub(r'\n+','\n', text).strip()

def split_md(md, file=None):
    md_header_splits = MD_SPLITTER.split_text(md)

    sections = split_docs(md_header_splits, ENCODING)

    documents, pages = [], []
    metas, contents = [], []
    for section in sections:
        meta, content = section.metadata.copy(), section.page_content
        metas.append(meta), contents.append(content)

    header = file.split(".")[0]
    if len(metas) <= 1:
        sections[0].metadata[1] = header
        pages = sections
    else:
        pages = combine_pages(contents=contents, metas=metas, header=header)
    # for i, doc in enumerate(pages):
    #     print(len(encoding.encode(doc.page_content)))
    # print('-------------------')
    # metadatas = [{f"header {k}":v for k, v in m.metadata.items()} for m in pages]
    metadatas = [(' | ').join([v for _, v in m.metadata.items()]) for m in pages]
    documents = [{'content': md_to_str(a.page_content, m), 'heading': m, 'markdown': a.page_content} for a, m in zip(pages, metadatas)]
    # with open(f'md_{file.split("/")[-1].split(".")[0]}.json', 'w') as f:
    #     f.write(json.dumps(doc_dict, indent=4))
    return documents

if __name__ == '__main__':
    import glob

    files = glob.glob('../md/*.md')
    files.sort()
    for file in files:
        md = open(file, 'r').read()
        split_md(md, file)

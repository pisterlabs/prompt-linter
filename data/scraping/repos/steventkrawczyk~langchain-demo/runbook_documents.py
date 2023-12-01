from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

source1 = """How to Handle a SegFault:

When the service encounters a segmentation fault, the proper solution is to restart the service. 
This can be done by providing the following command to the VM: 

```
sudo service restart
```
"""

doc1 = Document(page_content=source1, metadata={"error": "segmentation fault"})

source2 = """How to Handle a Timeout:

When the service hits timeouts, we need to investigate whether or not there is enough space left on the device.
To do this, we need to run the following command:

```
sudo df
```
"""

doc2 = Document(page_content=source2, metadata={"error": "timeout"})

source3 = """How to Handle a Disk Error:

Disk errors mean the entire VM has reached a bad state. We need to reboot the entire VM by running this command: 

```
sudo reboot
```
"""

doc3 = Document(page_content=source3, metadata={"error": "disk error"})

DOCS = [doc1, doc2, doc3]

def create_index():
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in DOCS:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

    search_index = Chroma.from_documents(source_chunks, OpenAIEmbeddings())
    return search_index
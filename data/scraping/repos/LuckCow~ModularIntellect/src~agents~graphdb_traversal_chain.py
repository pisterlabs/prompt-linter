"""
GraphDBTraversalChain allows an LLM to explore context chunks and their connections in a graph database,
extract relevant information, and continue exploring the graph database for additional information.
This helps the LLM to process and summarize the information obtained from the
database and also determine whether further exploration is needed.
"""
import re

from langchain import PromptTemplate
from langchain.schema import Document, SystemMessage
from langchain.chains import LLMChain
from langchain.chains.base import Chain

from typing import Dict, List

from langchain.vectorstores import VectorStore
from pydantic import PrivateAttr


# gives instructions for how to use memory system
mem_system_message = """Act as a researcher trying to answer a question about a topic. You have access to a knowledge base of documents broken down into context chunks that are connected to eachother. 
Information will be presented to you with a small number of chunks at a time. Each chunk of information may or may not be relevant to the question.
Additionally, each context chunk also has connection references to other chunks that are not immediately presented, but can be accessed by requesting to lookup the chunk referenced by the connection.
For each chunk that you are presented with, list out the chunk number, then evaluate whether or not the chunk is relevant to the question. If the chunk is relevant, provide a summary of the relevant information in the chunk, otherwise, just put 'None'
Then, give the best answer you can with the information that you have so far.
Finally, reflect on that answer and decide if it can be improved by looking up additional information from the connections. If so, list out any of the connections that you think would help improve the answer.
For example, if you are presented a chunk that seems like it is about to have important information to answer the question,
 but stops and has a connection that "CONTINUES TO" another chunk, you can respond by saying: "LOOKUP CONTEXT CHUNK #<chunk id>".
For example, your answer should follow this format: 
"Chunk Review: 
Chunk #<chunk id> - Relevant - <summary of relevance information in chunk>
Chunk #<chunk id> - Not Relevant - None
Chunk #<chunk id> - Relevant - <summary of relevance information in chunk>
Answer so far: <answer>
Further exploration chunk connection lookups: 
CONTINUES TO Context Chunk #<chunk id>
CONTINUES TO Context Chunk #<chunk id>
"
"""

# Presents context information
mem_query_template = PromptTemplate.from_template("""The question is: 
QUESTION: {question}

Here is the information gathered so far:
WORKING SUMMARY: {working_summary}

Below are the relevant context chunks that have been looked up so far:
CONTEXT CHUNKS:
{context_chunks}""")


class GraphDBTraversalChain(Chain):
    """
    give llm context chunks and their connections within a graph db and
    ask the llm to pick out relevant information and continue to explore the graph db for more information

    Attributes:
        llm_chain: LLMChain used to query LLM
        graph_vector_store: VectorStore instance that is used to retrieve context chunks based on the input.
        max_depth: The maximum depth to traverse in the graph database (default is 3).
        starting_chunks: The number of initial context chunks to provide to the LLM (default is 4).
    """
    llm_chain: LLMChain = None
    graph_vector_store: VectorStore
    max_depth: int = 3
    starting_chunks: int = 4

    _document_map_id_to_num: Dict[str, str] = PrivateAttr()
    _document_map_num_to_id: Dict[str, str] = PrivateAttr()

    @property
    def input_keys(self) -> List[str]:
        return ['input']

    @property
    def output_keys(self) -> List[str]:
        return ['output']

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._document_map_id_to_num = {}  # maps db id to local context number
        self._document_map_num_to_id = {}  # maps local context number to db id


    def explain_document_graph(self, documents: list[Document]) -> str:
        """
        Create a plain text description of a documents from graph database and their connections
        to feed context to llm

        Output format:
        Document Chunk#<num>
        Content: <all of the content of the document here>
        Connections:
            CONTINUES to Document Chunk #<num>
            CONTINUES FROM Document Chunk #<num>
        """

        description = []

        for document in documents:

            # Document ID and content
            description.append(f"Context Chunk #{self.map_id_to_number(document.metadata['id'])}")
            description.append(f"Content: {document.page_content}")

            # Document connections
            description.append("Connections:")

            connections = document.metadata.get('connections', [])

            for connection in connections:
                if 'type' in connection and 'connected_id' in connection and 'direction' in connection:
                    direction = "TO" if connection['direction'] == "out" else "FROM"
                    conn_num = self.map_id_to_number(connection['connected_id'])
                    description.append(f"    {connection['type']} {direction} Document Chunk #{conn_num}")

            description.append("\n")

        return "\n".join(description)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        # Look up relevant context chunks from vector store
        context_docs = self.graph_vector_store.search(inputs['input'], search_type='similarity', k=self.starting_chunks)

        # Create a mapping from db id to local context number
        for doc in context_docs:
            self.add_to_document_map(doc)

        working_summary = "None"
        depth = 0

        while True:

            output = self.llm_chain({'question': inputs['input'],
                                     'working_summary': working_summary,
                                     'context_chunks': self.explain_document_graph(context_docs)})
            # Parse output to get working summary
            output.update(_parse_output(output['response']))
            working_summary = output['working_summary']

            # Stop calling once no additional exploration calls are made or the max depth is reached
            depth += 1
            if depth >= self.max_depth or not output['requested_connections']:
                break

            # Look up the requested connections
            context_docs = []
            for connection in output['requested_connections']:
                # dereference local context number to db id
                connection_id = self.map_num_to_id(connection['num'])
                doc = self.graph_vector_store.docstore.search(connection_id)
                self.add_to_document_map(doc)
                context_docs.append(doc)

        return {'output': output}

    def add_to_document_map(self, doc):
        # add document
        self._add_to_document_map(doc.metadata['id'])

        # add connections
        for connection in doc.metadata.get('connections', []):
            self._add_to_document_map(connection['connected_id'])

    def _add_to_document_map(self, doc_id):
        # check if document is already in the map
        if doc_id in self._document_map_id_to_num:
            return
        doc_num = str(len(self._document_map_id_to_num) + 1)
        self._document_map_id_to_num[doc_id] = doc_num
        self._document_map_num_to_id[doc_num] = doc_id

    def map_id_to_number(self, doc_id):
        return self._document_map_id_to_num[doc_id]

    def map_num_to_id(self, num):
        return self._document_map_num_to_id[num]


def _parse_output(output):
    """
    Parse the output of the llm chain to get the working summary, requested connections, and reviewed chunks
    returns chunks analysis, working summary and list of context chunks that were looked up
    """
    # Sections within the output that we want to parse
    markers = ['Chunk Review:', 'Answer so far:', 'Further exploration chunk connection lookups:']

    # Escape the markers for regex use
    markers = [re.escape(marker) for marker in markers]

    # Create a pattern that matches the markers
    pattern = '|'.join(markers)

    # Split the output using the pattern
    parts = re.split(pattern, output)

    # Skip the first part, because it's before the first marker
    _, chunk_review, summary, connections = parts

    # strip whitespace
    summary = summary.strip()

    # Parse the chunk review section
    chunk_data = []
    for m in re.finditer(r'Chunk #(\d+) - (Relevant|Not Relevant) - (.*)', chunk_review):
        chunk_data.append({'num': m.group(1),
                       'relevant': True if m.group(2) == 'Relevant' else False,
                       'summary': m.group(3)})

    # Parse the connections section
    connection_data = []
    for m in re.finditer(r'(\w+) TO Context Chunk #(\d+)', connections):
        connection_data.append({'num': m.group(2),
                       'connection_type': m.group(1)})


    return {'reviewed_chunks': chunk_data, 'working_summary': summary, 'requested_connections': connection_data}


if __name__ == '__main__':
    from langchain.callbacks import StdOutCallbackHandler
    from langchain.chat_models import ChatOpenAI

    from src.agents.chat_chain import ChatChain
    from src.memory.triple_modal_memory import TripleModalMemory
    import os
    from dotenv import load_dotenv

    # Set up the cache
    import langchain
    from langchain.cache import SQLiteCache

    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

    # initialize the memory
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    mem = TripleModalMemory(uri, user, password)

    # Create memory from docks or load from file if it exists
    ingested = os.path.exists('../data/triple_modal_memory.faiss')
    if not ingested:
        knowledge_path = r'C:\Users\colli\Documents\AIPapers'
        mem.ingest_docs(knowledge_path)
        mem.save()
        print("Memory initialized and saved.")

    else:
        mem.load()
        print("Memory loaded.")

    handler = StdOutCallbackHandler()

    llm = ChatOpenAI(
        model_name="gpt-4",  # "gpt-3.5-turbo"
        temperature=0,
        verbose=True
    )
    chain = ChatChain(llm=llm, prompt=mem_query_template, callbacks=[handler], system_message=mem_system_message)
    knowledge_base_query_agent = GraphDBTraversalChain(llm_chain=chain, graph_vector_store=mem.vector_store)


    # Example Research questions:
    # What are different methods of providing language models with additional context to better answer questions?
    # How can semantic search be used in conjunction with large language models in order to better answer questions?
    # What are some techniques for achieving better general intelligence in language models?

    def main_loop():
        try:
            while True:
                question = input("Enter a question: ")
                print(knowledge_base_query_agent.run(question))


        except KeyboardInterrupt:
            print("Shutdown: Saving...")
            mem.save()
            print("Shutdown: Complete")

        else:
            print("Completed all tasks.")

"""
Example:-------------------------------------------------
What are the known uses of Glowing Moon Berries?

Context Chunk #1
Content: 'Glowing Moon Berries are a rare type of berry found only on the moon Zaphiron in the Pegasus galaxy. These luminescent berries shine brightly in the moon's perennial twilight, giving them their distinctive name.'
Connections:
    CONTINUES TO Context Chunk #3
    CONTINUES FROM Context Chunk #4

Context Chunk #2
Content: 'Glowing Moon Berries have a bitter, almost electric taste and little nutrition value. They are not considered edible by most species, and have been largely ignored until some interesting uses were discovered recently.'
Connections:
    CONTINUES TO Context Chunk #5
    CONTINUES FROM Context Chunk #6

Context Chunk #7
Content: 'Nebula Nectar is an extraordinary substance, harvested from the heart of the Orion Nebula. Nebula Nectar resembles Glowing Moon Berries in some ways which is interesting given how far apart the two galaxies are.'
Connections:
    CONTINUES TO Context Chunk #8
    CONTINUES FROM Context Chunk #9

----
Document #5
Content: 'Glowing Moon Berries are known for their unique properties. They are used primarily as a power source for the nano-tech machinery on Zaphiron due to their unusually high energy output.'
Connections:
    CONTINUES to Document #12
    CONTINUES FROM Document #2

Document #3
Content: 'In 2225, during the maiden voyage of the interstellar exploration ship 'Star Wanderer', the crew made a surprising discovery on Zaphiron, an obscure moon in the Pegasus galaxy. Amidst the constant twilight of the moon's surface, they found clusters of luminescent berries, later named 'Glowing Moon Berries', their radiance illuminating the alien landscape with an ethereal glow.'
Connections:
    CONTINUES to Document #15
    CONTINUES FROM Document #1

Document #15
Content: 'Later in 2225, as the Star Wanderer's crew continued their exploration of Zaphiron, they uncovered ancient ruins of a long-lost civilization, revealing intricate carvings that eerily mirrored constellations observed from Earth. This discovery deepened the mystery of Zaphiron, hinting at a potential connection between this distant moon and our home planet'
Connections:
    CONTINUES to Document #16
    CONTINUES FROM Document #15

Document #12
Content: 'Remarkably, when juiced and refined, the berries can power the nano-tech machines for months on a single serving, outperforming any other known energy source. Furthermore, the energy they emit is clean and sustainable, making Glowing Moon Berries a crucial component in the maintenance of the delicate ecological balance on Zaphiron.'
Connections:
    CONTINUES to Document #17
    CONTINUES FROM Document #5
"""
from langchain.graphs import GraphStore, GraphDocument, Node, Relationship
from langchain.chains import create_tagging_chain
from langchain.chat_models import ChatOpenAI

class GraphStoreSystem:
    def __init__(self):
        self.graph_store = GraphStore()
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    def add_code(self, code, metadata=None, schema=None):
        # Create a GraphDocument to hold the nodes and relationships
        doc = GraphDocument()
        # Create a Node for the code snippet
        code_node = Node(properties={'code': code})
        doc.add_node(code_node)

        # If there's metadata, create nodes and relationships for each metadata item
        if metadata:
            for key, value in metadata.items():
                metadata_node = Node(properties={key: value})
                doc.add_node(metadata_node)
                relationship = Relationship(start_node=code_node, end_node=metadata_node, type=key)
                doc.add_relationship(relationship)

        # If there's a schema, create a tagging chain and run it to generate tags
        if schema:
            chain = create_tagging_chain(schema, self.llm)
            tags = chain.run(code)
            # Create nodes and relationships for each tag
            for key, value in tags.items():
                tag_node = Node(properties={key: value})
                doc.add_node(tag_node)
                relationship = Relationship(start_node=code_node, end_node=tag_node, type=key)
                doc.add_relationship(relationship)

        # Save the GraphDocument to the GraphStore
        self.graph_store.add_graph_documents([doc])

# Usage:
gss = GraphStoreSystem()
schema = {
    "properties": {
        "sentiment": {"type": "string"},
        "aggressiveness": {"type": "integer"},
        "language": {"type": "string"},
    }
}
gss.add_code("some code snippet", schema=schema)

from dataclasses import dataclass
import json
import openai
import pinecone
api_key = "5837c25c-f54b-4dfc-9e4a-f00b694c0f67"
environment = "us-west4-gcp"
name = "name"
pinecone.init(api_key=api_key, environment=environment)

@dataclass
class Func:
    label: str
    description: str  # same as label for now
    type: str  # custom or built-in
    namespace: str  # operator or function


@dataclass
class Var:
    label: str
    description: str
    type: str  # custom or built-in
    dataType: str


class DB():
    def __init__(self, api_key, environment):
        openai.api_key = "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"
        self.api_key = api_key
        self.environment = environment
        pinecone.init(api_key=self.api_key, environment=self.environment)

    def initDB(self):
        pinecone.init(api_key=self.api_key, environment=self.environment)

    def createIndex(self, name):
        index = pinecone.create_index(name, 1536)
        return index

    def get_index(self, name):
        return pinecone.Index(name)

    def addFunction(self, function: Func):
        index = self.get_index("name")
        metadata = {
            "label": function.label,
            "description": function.description,
            "type": function.type,
            # "oper": function.oper 
        }
        vector = self.getEmbedding(function.description)
        index.upsert(
            vectors=[(function.label, vector, metadata)],
            namespace=function.namespace,
        )

    def addFunctions(self, functions: list[Func]):
        for func in functions:
            self.addFunction(func)

    def addVariable(self, variable: Var):
        index = self.get_index("name")
        metadata = {
            "label": variable.label,
            "description": variable.description,
            "type": variable.type,
            "dataType": variable.dataType
        }
        vector = self.getEmbedding(variable.description)
        index.upsert(
            vectors=[(variable.label, vector, metadata)],
            namespace="variables",
        )

    def addVariables(self, variables: list[Var]):
        for var in variables:
            self.addVariable(var)

    def getVariable(self, name):
        index = self.get_index("name")
        filtered = index.query(
            vector=self.getEmbedding(name), 
            top_k=5, 
            namespace="variables",
            include_metadata=True,)
        # print(filtered)
        return filtered.matches[0].metadata

    def getFunction(self, name, namespace="functions"):
        index = self.get_index("name")
        filtered = index.query(
            vector=self.getEmbedding(name), 
            top_k=1, 
            namespace=namespace,
            include_metadata=True)
        return filtered.matches[0].metadata
        print(filtered)

    def getEmbedding(self, phrase):
        response = openai.Embedding.create(
            input=[phrase],
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']


def loadColumns(path: str):
    # load the columns from a file
    with open(path, 'r') as f:
        columns = json.load(f)
    return [
        Var(
            label=column['lable'],
            description=column['description'],
            type="standard",
            dataType="float"
        )
        for column in columns
        if len(column.keys()) == 3
    ]

def loadFunctions(path: str, types: str, namespace: str):
    with open(path, 'r') as f:
        functions = json.load(f)

    return [
        Func(
            label=function['saytex'],
            description=function['saytex'],
            type=types,
            namespace=namespace
        )
        for function in functions
    ]
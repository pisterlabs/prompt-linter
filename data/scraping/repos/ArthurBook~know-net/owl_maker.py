import os
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from know_net.graph_building import LLMGraphBuilder

import pickle

ONT = """@prefix : <http://www.semanticweb.org/ontologies/technology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

: a owl:Ontology .

:Technology a owl:Class .

:Hardware a owl:Class ;
    rdfs:subClassOf :Technology .

:Software a owl:Class ;
    rdfs:subClassOf :Technology .

:AI a owl:Class ;
    rdfs:subClassOf :Software .

:MachineLearning a owl:Class ;
    rdfs:subClassOf :AI .

:DeepLearning a owl:Class ;
    rdfs:subClassOf :MachineLearning .

:Robotics a owl:Class ;
    rdfs:subClassOf :Hardware .

:Drone a owl:Class ;
    rdfs:subClassOf :Robotics .

:Computer a owl:Class ;
    rdfs:subClassOf :Hardware .

:Smartphone a owl:Class ;
    rdfs:subClassOf :Hardware .

:OperatingSystem a owl:Class ;
    rdfs:subClassOf :Software .

:Linux a owl:Class ;
    rdfs:subClassOf :OperatingSystem .

:Windows a owl:Class ;
    rdfs:subClassOf :OperatingSystem .

:MacOS a owl:Class ;
    rdfs:subClassOf :OperatingSystem .

:Android a owl:Class ;
    rdfs:subClassOf :OperatingSystem .

:iOS a owl:Class ;
    rdfs:subClassOf :OperatingSystem .

:Person a owl:Class .

:Developer a owl:Class ;
    rdfs:subClassOf :Person .

:Researcher a owl:Class ;
    rdfs:subClassOf :Person .

:CEO a owl:Class ;
    rdfs:subClassOf :Person .

:Company a owl:Class .

:Startup a owl:Class ;
    rdfs:subClassOf :Company .

:Multinational a owl:Class ;
    rdfs:subClassOf :Company .

:Innovation a owl:Class .

:Patent a owl:Class ;
    rdfs:subClassOf :Innovation .

:ResearchPaper a owl:Class ;
    rdfs:subClassOf :Innovation .

:News a owl:Class .

:BlogPost a owl:Class ;
    rdfs:subClassOf :News .

:PressRelease a owl:Class ;
    rdfs:subClassOf :News .

:Conference a owl:Class .

:Webinar a owl:Class ;
    rdfs:subClassOf :Conference .

:Seminar a owl:Class ;
    rdfs:subClassOf :Conference .

:Product a owl:Class .

:SoftwareProduct a owl:Class ;
    rdfs:subClassOf :Product .

:HardwareProduct a owl:Class ;
    rdfs:subClassOf :Product .

:Service a owl:Class .

:CloudService a owl:Class ;
    rdfs:subClassOf :Service .

:ConsultingService a owl:Class ;
    rdfs:subClassOf :Service .

:Investment a owl:Class .

:VentureCapital a owl:Class ;
    rdfs:subClassOf :Investment .

:Acquisition a owl:Class ;
    rdfs:subClassOf :Investment .

:Regulation a owl:Class .

:PrivacyPolicy a owl:Class ;
    rdfs:subClassOf :Regulation .

:DataProtection a owl:Class ;
    rdfs:subClassOf :Regulation .

:worksFor a owl:ObjectProperty ;
    rdfs:domain :Person ;
    rdfs:range :Company .

:develops a owl:ObjectProperty ;
    rdfs:domain :Developer ;
    rdfs:range :Product .

:investsIn a owl:ObjectProperty ;
    rdfs:domain :VentureCapital ;
    rdfs:range :Startup .

:publishes a owl:ObjectProperty ;
    rdfs:domain :Researcher ;
    rdfs:range :ResearchPaper .

:attends a owl:ObjectProperty ;
    rdfs:domain :Person ;
    rdfs:range :Conference .

:owns a owl:ObjectProperty ;
    rdfs:domain :Company ;
    rdfs:range :Product .

:regulates a owl:ObjectProperty ;
    rdfs:domain :Regulation ;
    rdfs:range :Company .
"""


def main() -> None:
    builder_path = os.environ["BUILDER_PKL_PATH"]
    with open(builder_path, "rb") as f:
        graph = pickle.load(f)
    graph: LLMGraphBuilder

    _PROMPT = """Given the following knowledge graph triples:
    triples: {triples}.
    Please extend the following Turtle OWL ontology.
    ontology: {ontology}.

    Your ontology should be purely additional on top of the above ontology.
    I should be able to simply append what you give me to the above ontology and load it directly into Protege.

    Output the result in JSON:
    {{"turtle": "value"}} with no other text please.
    """
    prompt = PromptTemplate(
        input_variables=["triples", "ontology"],
        template=_PROMPT,
    )
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    chain = LLMChain(llm=llm, prompt=prompt)

    def create_batches(long_list, bs: int = 16):
        batch_size = bs
        num_batches = len(long_list) // batch_size
        remainder = len(long_list) % batch_size

        batches = []
        start_index = 0

        for i in range(num_batches):
            end_index = start_index + batch_size
            batch = long_list[start_index:end_index]
            batches.append(batch)
            start_index = end_index

        if remainder != 0:
            last_batch = long_list[-remainder:]
            batches.append(last_batch)

        return batches

    import json

    for i, nodes in enumerate(create_batches(graph.triples)):
        print(i)
        try:
            res = chain.predict(triples=nodes, ontology=ONT)
            t = res.replace("\n", "\\n")
            d = json.loads(t)
            turtle = d["turtle"]
            with open(f"data/turtle_{i}", "w") as f:
                f.write(turtle)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()

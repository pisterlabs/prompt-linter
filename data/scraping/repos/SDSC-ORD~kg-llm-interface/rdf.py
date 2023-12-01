# kg-llm-interface
# Copyright 2023 - Swiss Data Science Center (SDSC)
# A partnership between École Polytechnique Fédérale de Lausanne (EPFL) and
# Eidgenössische Technische Hochschule Zürich (ETHZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import groupby
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from langchain.schema import Document
from rdflib import ConjunctiveGraph, Graph
from rdflib.exceptions import ParserError
from SPARQLWrapper import SPARQLWrapper, CSV
from urllib.parse import urlparse

# Retrieve triples of human readable labels/values from a SPARQL endpoint.
TRIPLE_LABEL_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?s ?p ?o ?sLab ?pLab ?oClean
WHERE
{{
    ?s ?p ?o .
    ?s rdfs:label ?sLab .
    ?p rdfs:label ?pLab .
    OPTIONAL {{
        ?o rdfs:label ?oLab .
        FILTER(LANG(?oLab) = "{lang}")
    }}
    BIND(COALESCE(?oLab, ?o) AS ?oLabOrUri)
    BIND(
        IF (isLiteral(?o), ?o, STR(?oLabOrUri))
        AS ?oLabOrVal
    )
    FILTER(LANG(?sLab) = "{lang}")
    FILTER(LANG(?pLab) = "{lang}")
    FILTER(LANG(?oLabOrVal) = "{lang}" || LANG(?oLabOrVal) = "")
    BIND (REPLACE(STR(?oLabOrVal), "^.*[#/:]([^/:#]*)$", "$1") as ?oClean)
    {graph_mask}
}}
"""

# Retrieve each subject and its annotations
SUBJECT_DOC_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?s ?sLab ?sCom
WHERE
{{
    ?s rdfs:label ?sLab .
    OPTIONAL {{
        ?s rdfs:comment ?sCom .
        ?o rdfs:label ?oLab .
    }}
    FILTER(LANG(?sLab) = "{lang}" || LANG(?sLab) = "")
    FILTER(LANG(?sCom) = "{lang}" || LANG(?sLab) = "")
    {graph_mask}
}}
"""


def make_graph_mask(graph: Optional[str] = None) -> str:
    if graph:
        return f"FILTER EXISTS {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}"
    else:
        return ""


def setup_kg(
    endpoint: str, user: Optional[str] = None, password: Optional[str] = None
) -> Graph | SPARQLWrapper:
    """Try to connect to SPARQL endpoint. If not a URL, attempt
    to parse RDF file with rdflib."""

    url = urlparse(endpoint)
    if url.scheme and url.netloc:
        kg = SPARQLWrapper(endpoint)
        kg.setReturnFormat(CSV)
        if user and password:
            kg.setCredentials(user, password)
    else:
        kg = ConjunctiveGraph()
        kg.parse(endpoint)
    return kg


def split_documents_from_endpoint(
    kg: Graph | SPARQLWrapper,
    graph: Optional[str] = None,
) -> Iterator[Document]:
    """Load subject-based documents from a SPARQL endpoint.

    Parameters
    ----------
    endpoint:
        URL of the SPARQL endpoint.
    user:
        Username to use for authentication.
    password:
        Password to use for authentication.
    graph:
        URI of named graph to load RDF data from.
        If not specified, all subjects are used.
    """

    graph_mask = make_graph_mask(graph)

    # Load the query results
    # Query results contain 6 columns:
    # subject, predicate, object, subject label, predicate label, object label
    results = query_kg(kg, TRIPLE_LABEL_QUERY.format(lang="en", graph_mask=graph_mask))
    # Exclude empty / incomplete results (e.g. missing labels)
    results = filter(lambda x: len(list(x)) == 6, results)
    next(results)  # skip header
    results = sorted(results, key=lambda x: x[0])[1:]
    # Yield triples and text by subject
    for k, g in groupby(results, lambda x: x[0]):
        # Original triples about subject k
        data = list(g)
        triples = "\n".join([f"<{s}> <{p}> <{o}>" for s, p, o, sl, pl, ol in data])
        # Human-readable "triples" about subject k
        doc = "\n".join([" ".join(elem[3:]) for elem in data])
        yield Document(page_content=doc, metadata={"subject": k, "triples": triples})


def get_subjects_docs(
    kg: Graph | SPARQLWrapper, graph: Optional[str] = None
) -> List[Document]:
    """Given an RDF graph, iterate over subjects, extract human-readable
    RDFS annotations. For each subject, retrieve a "text document" with
    original triples attached as metadata."""

    results = query_kg(
        kg, SUBJECT_DOC_QUERY.format(lang="en", graph_mask=make_graph_mask(graph))
    )
    docs = []

    # Skipping header row
    if isinstance(kg, SPARQLWrapper):
        results = results[1:]

    for sub, label, comment in results:
        if comment is None:
            comment = ""
        text = f"""
        {label}
        {comment}
        """
        triples = query_kg(kg, f"DESCRIBE <{sub}>")

        g = Graph()
        # SPARQLWrapper returns a ntriple string, rdflib a list of triples
        try:
            g.parse(data=triples[0][0], format="nt")
        except (RuntimeError, ParserError):
            for triple in triples:
                g.add(triple)
        meta = {"triples": g.serialize(format="nt")}
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def query_kg(kg: Graph | SPARQLWrapper, query: str) -> List[List[Any]]:
    """Query a knowledge graph, either an rdflib Graph or a SPARQLWrapper.
    Results are returned as a list of lists representing a table."""
    query2fmt = {"DESCRIBE": "nt", "SELECT": "csv", "CONSTRUCT": "nt"}
    if isinstance(kg, Graph):
        resp = kg.query(query)
        fmt = query2fmt[resp.type]
        raw_results = resp.serialize(format=fmt)

    elif isinstance(kg, SPARQLWrapper):
        kg.setQuery(query)
        fmt = query2fmt[kg.queryType]
        kg.setReturnFormat(fmt)
        raw_results = kg.query().convert()
    else:
        raise ValueError(f"Invalid type for kg: {type(kg)}")
    if fmt == "csv":
        import csv

        lines = raw_results.decode("utf-8").splitlines()
        return [row for row in csv.reader(lines, quotechar='"', delimiter=",") if row]
    else:
        return [[raw_results]]

    return results

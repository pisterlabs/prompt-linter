from fastapi import HTTPException
import psycopg2
from pydantic import BaseModel
from unstructured.partition.html import partition_html
from langchain.graphs.graph_document import GraphDocument, Node, Relationship
from langchain.schema import Document
import spacy
from spacy import displacy

import re
from dataclasses import dataclass
import json
import itertools
import logging

from research_copilot.llm import get_chat_completion
from research_copilot.db.graph import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")


@dataclass
class Section:
    id: str
    law_code: str
    section_num: str
    division: str
    title: str
    part: str
    chapter: str
    article: str
    content_xml: str


def section_to_graph_doc(section):
    source = Document(page_content=section.content_xml, metadata=section.__dict__)
    # TODO move all partitioning/cleaning/combining to separate function
    elements = partition_content_xml(section.content_xml)
    nodes = []
    relationships = []
    extracted_paragraphs = map(
        extract_text_between_parentheses, [e.to_dict()["text"] for e in elements]
    )
    cleaned_paragraphs = []
    for i, (p_id, p_text) in enumerate(extracted_paragraphs):
        if p_id or i == 0:
            cleaned_paragraphs.append((p_id, clean_extra_whitespace(p_text)))
            continue
        logger.info(
            f"Could not extract id from {p_text[:20]} in section {section.id}. Combining!"
        )
        cleaned_paragraphs[-1] = (
            cleaned_paragraphs[-1][0],
            cleaned_paragraphs[-1][1] + "\n" + clean_extra_whitespace(p_text),
        )

    # Graph construction
    section_props = {
        "law_code": section.law_code,
        "section_num": section.section_num,
        "division": section.division,
        "title": section.title,
        "part": section.part,
        "chapter": section.chapter,
        "article": section.article,
    }
    path_stack = []
    for i, (p_id, p_text) in enumerate(cleaned_paragraphs):
        # TODO revisit adding reference edges while crawling graph
        # p_sentences = p_text.split(". ")
        # p_sec_mentions = list(
        #     itertools.chain(*map(extract_section_mentions, p_sentences))
        # )
        # p_sec_refs = create_sec_refs(p_sec_mentions)
        if not path_stack:
            # Create top level section node
            section_has_top_level_text = p_id is None
            section_node = Node(
                id=section.id,
                type="Section",
                properties={
                    **section_props,
                    **({"text": p_text} if section_has_top_level_text else {}),
                },
            )
            nodes.append(section_node)
            path_stack.append(section_node)
            if section_has_top_level_text:
                continue
        previous_pid = nodes[-1].properties.get("p_id")
        parent_pid = path_stack[-1].properties.get("p_id")

        # Change level if necessary
        if subsection_level_type(p_id) == subsection_level_type(parent_pid):
            path_stack.pop()
        elif subsection_level_type(p_id) != subsection_level_type(previous_pid):
            path_stack.append(nodes[-1])

        subsection_node = Node(
            id=get_subsection_id(path_stack[-1].id, p_id),
            type="Subsection",
            properties={
                **section_props,
                "p_id": p_id,
                "text": p_text or "",  # I think these both get nulled anyway
            },
        )
        relationships.append(
            Relationship(source=nodes[-1], target=subsection_node, type="FOLLOWED_BY")
        )
        relationships.append(
            Relationship(
                source=path_stack[-1],
                target=subsection_node,
                type="PARENT_OF",
            )
        )
        nodes.append(subsection_node)
    logger.info(
        f"Adding graph document to graph: {len(nodes)} nodes, {len(relationships)} relationships."
    )
    graph_doc = GraphDocument(nodes=nodes, relationships=relationships, source=source)
    return graph_doc


def subsection_level_type(p_id):
    if p_id is None:
        return -1
    if re.search(r"[a-z]+", p_id):
        return 0
    elif re.search(r"[0-9]+", p_id):
        return 1
    elif re.search(r"[A-Z]+", p_id):
        return 2
    return 3


def get_subsection_id(parent_id, id):
    return f"{parent_id}_{id}"


def get_sections_from_db(code="CIV", ccpa_only=False):
    with psycopg2.connect(
        host="localhost", database="capublic", user="taylor", password="postgres"
    ) as conn:
        # create cursor
        with conn.cursor() as cur:
            query = (
                """
            select 
              id, law_code, section_num, division, title, part, chapter, article, content_xml
             from law_section_tbl where law_code = %s;
            """
                if not ccpa_only
                else f"""
            select 
              id, law_code, section_num, division, title, part, chapter, article, content_xml
             from law_section_tbl 
             where law_code = 'CIV' and division = '3.' and part = '4.' and title = '1.81.5.' ;
            """
            )

            # execute query
            cur.execute(query, (code,))

            # fetch all results
            results = cur.fetchall()
            sections = [Section(*result) for result in results]
            return sections


def load_sections(sections):
    logger.info(f"Loading {len(sections)} sections into graph db!")
    # graph_docs = []
    for section in sections:
        gd = section_to_graph_doc(section)
        graph.add_graph_documents([gd], include_source=False)
        # FIXME for some reason this subtly breaks by dropping props from nodes... maybe at limit of query bandwidth?
        # if graph_docs:
        #     graph_docs[-1].relationships.append(
        #         Relationship(
        #             source=graph_docs[-1].nodes[-1],
        #             target=gd.nodes[0],
        #             type="FOLLOWED_BY",
        #         )
        #     )
        # graph_docs.append(gd)
    # graph.add_graph_documents(graph_docs, include_source=False)
    apply_codepiece_label()
    embed_all_codepiece_node_texts()
    logger.info(f"Finished loading {len(sections)} sections into graph db!")


def load_code(code):
    logger.info(f"Loading code {code} into graph db!")
    sections = get_sections_from_db(code=code)
    load_sections(sections)
    logger.info(f"Finished loading code {code} into graph db!")


def partition_content_xml(content_xml):
    elements = partition_html(text=content_xml)
    return elements


def extract_text_between_parentheses(text):
    match = re.match(r"^\((.*?)\)(.*)", text)
    if match:
        return match.group(1), match.group(2)
    else:
        return None, text


def clean_extra_whitespace(text: str) -> str:
    """Cleans extra whitespace characters that appear between words.

    Example
    -------
    ITEM 1.     BUSINESS -> ITEM 1. BUSINESS
    """
    cleaned_text = re.sub(r"[\xa0\n\t]", " ", text)  # include tab
    cleaned_text = re.sub(r"([ ]{2,})", " ", cleaned_text)
    return cleaned_text.strip()


def resolve_sec_ref(unresolved_sec_ref: dict, current_node_sec_ref: dict):
    if unresolved_sec_ref.get("code") and unresolved_sec_ref.get(
        "code"
    ) != current_node_sec_ref.get("code"):
        return unresolved_sec_ref
    return {
        **current_node_sec_ref,
        **unresolved_sec_ref,
    }


def extract_section_mentions(text):
    section_terms = ["section", "division", "titile", "part", "chapter", "article"]
    subsection_terms = ["clause", "paragraph", "subdivision"]
    code_subpatterns = [r"code"]
    section_subpatterns = [rf"{term}s?\s+[\d.]+" for term in section_terms]
    subsection_subpatterns = [rf"{term}s?\s+\(\w+\)" for term in subsection_terms]
    all_subpatterns = code_subpatterns + section_subpatterns + subsection_subpatterns

    # FIXME why doesn't it work if I don't sentence split first?
    doc = nlp(text)
    section_texts = []
    for sent in doc.sents:
        pattern = rf"(?i)({'|'.join(all_subpatterns)})"
        matches = re.finditer(pattern, sent.text)

        displacy.render(sent, style="dep", options={"compact": True})

        spans = [sent.char_span(*m.span()) for m in matches]

        def span_for_subtree(subtree):
            tokens = list(subtree)
            return sent[tokens[0].i : tokens[-1].i + 1]

        subtree_spans = spacy.util.filter_spans(
            [span_for_subtree(span.subtree) for span in spans if span is not None]
        )
        for span in subtree_spans:
            if span is not None and len(span) > 0:
                subtree_text = "".join(
                    [token.text + token.whitespace_ for token in span.subtree]
                )
                section_texts.append(subtree_text)
                print(subtree_text)
    return section_texts


def create_sec_refs(sec_mentions):
    CODES = {
        "BPC": "Business and Professions Code - BPC",
        "CCP": "Code of Civil Procedure - CCP",
        "CIV": "Civil Code - CIV",
        "COM": "Commercial Code - COM",
        "CORP": "Corporations Code - CORP",
        "EDC": "Education Code - EDC",
        "ELEC": "Elections Code - ELEC",
        "EVID": "Evidence Code - EVID",
        "FAM": "Family Code - FAM",
        "FIN": "Financial Code - FIN",
        "FGC": "Fish and Game Code - FGC",
        "FAC": "Food and Agricultural Code - FAC",
        "GOV": "Government Code - GOV",
        "HNC": "Harbors and Navigation Code - HNC",
        "HSC": "Health and Safety Code - HSC",
        "INS": "Insurance Code - INS",
        "LAB": "Labor Code - LAB",
        "MVC": "Military and Veterans Code - MVC",
        "PEN": "Penal Code - PEN",
        "PROB": "Probate Code - PROB",
        "PCC": "Public Contract Code - PCC",
        "PRC": "Public Resources Code - PRC",
        "PUC": "Public Utilities Code - PUC",
        "RTC": "Revenue and Taxation Code - RTC",
        "SHC": "Streets and Highways Code - SHC",
        "UIC": "Unemployment Insurance Code - UIC",
        "VEH": "Vehicle Code - VEH",
        "WAT": "Water Code - WAT",
        "WIC": "Welfare and Institutions Code - WIC",
        "CONS": "California Constitution - CONS",
    }

    prompt = f"""
  You will be provided with a list of natural language references to sections of the law. You must represent each reference as a json object with as many of the fields as you can surmise.
  Here are valid values for the code field, with discriptions: {CODES}
  If there is a code but the code is not listed in the above list of valid values, put the plain text.

  Some examples:

  Reference: ["paragraph (7) of subdivision (a) of Section 1798.185 in Division 3, Part 4, Title 9 of the Civil Code of California"]
  Result: [{{
    'code': 'CIV',
    'division': '3',
    'part': '4',
    'title': '9',
    'section': '1798.188',
    'subsections': ['a', '7'],
  }}]

  Reference: ["paragraph (5) of subdivision (b) of Section 1798.190", "clause (ii)"]
  Result: [{{
    'section': '1798.190',
    'subsections': ['b', '5'],
  }}, {{
    'subsections': ['ii'],
  }}]

  Reference: ["Section 17014 of Title 18 of the California Code of Regulations"]
  Result: [{{
    'code': 'California Code of Regulations',
    'title': '18',
    'section': '17014',
  }}]

  Now you try;

  Reference: {sec_mentions}
  Result: 
  """

    return json.loads(get_chat_completion(prompt))


from sentence_transformers import SentenceTransformer


def create_codepiece_embedding_index(model, type="node", measure="cosine"):
    embedding_name = get_embedding_name(model, type)
    embedding_model = SentenceTransformer(model)
    dimension = embedding_model.get_sentence_embedding_dimension()
    q = """
    CALL db.index.vector.createNodeIndex($index_name, 'CodePiece', $embedding_name, $dimension, $measure)
    """
    graph.query(
        q,
        params={
            "embedding_name": embedding_name,
            "index_name": embedding_index_name(embedding_name),
            "dimension": dimension,
            "measure": measure,
        },
    )


def embed_all_codepiece_node_texts(model="thenlper/gte-large"):
    embedding_name = get_embedding_name(model)
    embedding_model = SentenceTransformer(model)
    # get all
    q = f"""
    MATCH (n:CodePiece)
    WHERE n.`{embedding_name}` IS NULL AND n.text IS NOT NULL AND n.text <> ''
    RETURN n.id, n.text
    """  # FIXME unsanitized query input
    results = graph.query(q)
    sentences = [r["n.text"] for r in results]
    logger.info(f"Embedding {len(sentences)} sentences with {model}")
    # embed all
    embeddings = embedding_model.encode(
        sentences, normalize_embeddings=False
    )  # TODO compare to normalized
    ids_embeddings = [(r["n.id"], e) for r, e in zip(results, embeddings)]
    logger.info(f"Updating {len(ids_embeddings)} codepiece nodes with embeddings")
    # set all
    q2 = """
    UNWIND $data AS pair
    MATCH (n:CodePiece)
    WHERE n.id = pair[0]
    CALL apoc.create.setProperty(n, $embedding_name, pair[1]) YIELD node
    RETURN NULL
    """
    graph.query(q2, params={"data": ids_embeddings, "embedding_name": embedding_name})
    logger.info(f"Embedded {len(ids_embeddings)} nodes.")


def joined_path_text(path_nodes):
    path_text = " | ".join([n.get("text", "") for n in path_nodes])
    return path_text


def embed_and_add_all_codepiece_node_path_texts(model="thenlper/gte-large"):
    """Create embeddings on the combined text of all parents and the codepiece node."""
    embedding_name = get_embedding_name(model, type="path")
    embedding_model = SentenceTransformer(model)
    # get all root paths
    # FIXME this is missing some sections with no children
    q = f"""
    MATCH p=(root:Section)-[r:PARENT_OF*]->(n:CodePiece) 
    WHERE n.`{embedding_name}` IS NULL AND n.text IS NOT NULL AND n.text <> '' AND NOT (n)-[:PARENT_OF]->()
    RETURN nodes(p) as path_nodes
    """  # FIXME unsanitized query input
    results = graph.query(q)

    path_texts = []
    for r in results:
        path_text = joined_path_text(r["path_nodes"])
        path_texts.append(path_text)
    logger.info(f"Embedding {len(path_texts)} sentences with {model}")
    # embed all
    embeddings = embedding_model.encode(
        path_texts, normalize_embeddings=False
    )  # TODO compare to normalized
    ids_embeddings_pathtexts = list(
        zip([r["path_nodes"][-1]["id"] for r in results], embeddings, path_texts)
    )
    logger.info(
        f"Updating {len(ids_embeddings_pathtexts)} codepiece nodes with embeddings"
    )
    # set all
    q2 = """
    UNWIND $data AS item
    MATCH (n:CodePiece)
    WHERE n.id = item[0]
    CALL apoc.create.setProperties(n, [$embedding_name, 'path_text'], [item[1], item[2]]) YIELD node
    RETURN NULL
    """
    graph.query(
        q2, params={"data": ids_embeddings_pathtexts, "embedding_name": embedding_name}
    )
    logger.info(f"Embedded {len(ids_embeddings_pathtexts)} nodes.")


def embedding_index_name(embedding_name, type="node"):
    return f"{embedding_name}_index"


def get_embedding_name(model, type="node"):
    if type == "node":
        return model.split("/")[-1]
    elif type == "path":
        return f"{model.split('/')[-1]}_path"
    raise ValueError(f"Invalid type {type}")


def apply_codepiece_label():
    q = """
    MATCH (n)
    WHERE n:Section OR n:Subsection
    SET n:CodePiece
    """
    graph.query(q)


def create_id_uniqueness_constraint():
    q = """
    CREATE CONSTRAINT unique_codepiece_id ON (n:CodePiece) ASSERT n.id IS UNIQUE
    """
    graph.query(q)


def remove_property_from_all_codepiece_nodes(property_name):
    q = f"""
    MATCH (n:CodePiece)
    CALL apoc.create.removeProperties(n, [$property_name]) YIELD node
    RETURN NULL
    """
    graph.query(q, params={"property_name": property_name})


def codepiece_text_similarity_search(
    text: str,
    embedding_model,
    embedding_name,
    ccpa_only=True,
    normalize_embeddings=False,
    limit=10,
):
    limit = min(limit, 15)
    embeddings = embedding_model.encode(
        [text], normalize_embeddings=normalize_embeddings
    )
    q = f"""
    CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
    YIELD node, score
    {"WHERE node.law_code = 'CIV' and node.division = '3.' and node.part = '4.' and node.title = '1.81.5.'" if ccpa_only else ""}
    RETURN node, score
    """
    results = graph.query(
        q,
        params={
            "index_name": embedding_index_name(embedding_name),
            "embedding": embeddings[0],
            "limit": limit,
        },
    )
    return list(sorted(results, key=lambda r: r["score"], reverse=True))


def get_path_nodes(node_ids):
    q = f"""
    UNWIND $data AS item
    MATCH p=(root:Section)-[r:PARENT_OF*]->(n:CodePiece) 
    WHERE n.id = item
    RETURN nodes(p) as path_nodes
    """
    return graph.query(q, params={"data": node_ids})


def codepiece_fulltext_search(
    text: str,
    ccpa_only=True,
    limit=10,
):
    limit = min(limit, 15)
    q = f"""
    CALL db.index.fulltext.queryNodes("codepiece_text_index", $text) YIELD node, score
    {"WHERE node.law_code = 'CIV' and node.division = '3.' and node.part = '4.' and node.title = '1.81.5.'" if ccpa_only else ""}
    RETURN node, score
    LIMIT $limit
    """
    results = graph.query(
        q,
        params={
            "text": text,
            "limit": limit,
        },
    )
    return list(sorted(results, key=lambda r: r["score"], reverse=True))


def create_codepiece_fulltext_index():
    q = """
    CREATE FULLTEXT INDEX codepiece_text_index FOR (n:CodePiece) ON EACH [n.text]
    OPTIONS {
    indexConfig: {
        `fulltext.eventually_consistent`: true
    }
    }
    """
    graph.query(q)


def retrieve_related_nodes(node, limit=10):
    limit = min(limit, 15)


def do_rag(embedding_model, query: str, params_dict: dict = {}):
    mode = params_dict.get("mode", "similarity")
    ccpa_only = params_dict.get("ccpa_only", True)
    limit = params_dict.get("limit", 5)
    related_limit = params_dict.get("related_limit", 0)

    if mode == "similarity":
        results = codepiece_text_similarity_search(
            text=query,
            embedding_model=embedding_model,
            embedding_name="gte-large",
            ccpa_only=ccpa_only,
            limit=limit,
        )
    elif mode == "path_similarity":
        results = codepiece_text_similarity_search(
            text=query,
            embedding_model=embedding_model,
            embedding_name="gte-large_path",
            ccpa_only=ccpa_only,
            limit=limit,
        )
        path_node_results = get_path_nodes([r["node"]["id"] for r in results])
        for i, r in enumerate(results):
            r["path_nodes"] = path_node_results[i]["path_nodes"]
        # related_limit = 5
    elif mode == "text":
        results = codepiece_fulltext_search(
            text=query,
            ccpa_only=ccpa_only,
            limit=limit,
        )
    else:
        raise HTTPException(status_code=442, detail="Invalid mode")

    if related_limit:
        for r in results:
            r["related_nodes"] = retrieve_related_nodes(
                r["node"], params_dict.get("related_limit", 5)
            )
    return results


class CodePathContext(BaseModel):
    leaf_id: str
    leaf_text: str
    parent_texts: list[str]


def get_all_codepiece_node_paths(ccpa_only=True):
    q = f"""
    MATCH p=(root:Section)-[r:PARENT_OF*]->(n:CodePiece) 
    WHERE n.text IS NOT NULL AND n.text <> '' AND NOT (n)-[:PARENT_OF]->()
    {"AND n.law_code = 'CIV' and n.division = '3.' and n.part = '4.' and n.title = '1.81.5.'" if ccpa_only else ""}
    RETURN nodes(p) as path_nodes
    UNION
    MATCH (s:Section)
    WHERE NOT (s)-[:PARENT_OF]->()
    {"AND s.law_code = 'CIV' and s.division = '3.' and s.part = '4.' and s.title = '1.81.5.'" if ccpa_only else ""}
    RETURN [s] as path_nodes
    """
    results = graph.query(q)
    return results


def update_codepiece_node_properties(node_id: str, props: dict):
    q = f"""
    MATCH (n:CodePiece)
    WHERE n.id = $node_id
    CALL apoc.create.setProperties(n, $prop_keys, $prop_values) YIELD node
    RETURN NULL
    """
    graph.query(
        q,
        params={
            "node_id": node_id,
            "prop_keys": list(props.keys()),
            "prop_values": list(props.values()),
        },
    )


def get_unfulfilled_codepiece_requirements(ccpa_only=True):
    # q = f"""
    # MATCH (n:CodePiece)
    # OPTIONAL MATCH (n)-[:IS_FULFILLED_BY]->(s:PrivacyPolicySection)
    # OPTIONAL MATCH (n)-[:IS_NOT_FULFILLED_BY]->(x:PrivacyPolicySection)
    # WHERE n.is_requirement = true
    # {"AND n.law_code = 'CIV' and n.division = '3.' and n.part = '4.' and n.title = '1.81.5.'" if ccpa_only else ""}
    # RETURN n as node, s as fulfilled_by, x as not_fullfilled_by
    # """
    #     q = f"""
    #     MATCH (n:CodePiece)
    #     WHERE n.is_requirement = true
    #     {"AND n.law_code = 'CIV' and n.division = '3.' and n.part = '4.' and n.title = '1.81.5.'" if ccpa_only else ""}
    #     WITH n
    #     OPTIONAL MATCH (n)-[:IS_FULFILLED_BY]->(s:PrivacyPolicySection)
    #     OPTIONAL MATCH (n)-[:IS_NOT_FULFILLED_BY]->(x:PrivacyPolicySection)
    #     RETURN n as node, s as fulfilled_by, x as not_fullfilled_by
    # """
    q = f"""
    MATCH (n:CodePiece)
    WHERE n.is_requirement = true
    AND NOT (n)-[:IS_FULFILLED_BY]->(:PrivacyPolicySection)
    {"AND n.law_code = 'CIV' and n.division = '3.' and n.part = '4.' and n.title = '1.81.5.'" if ccpa_only else ""}
    RETURN n as node
"""
    return graph.query(q)


def mark_fulfilled_by(req_id, section_id):
    # q = """
    # MATCH (n:CodePiece)
    # WHERE n.id = $req_id AND n.is_requirement = true
    # MATCH (s:PrivacyPolicySection)
    # WHERE s.id = $section_id
    # MERGE (n)-[:IS_FULFILLED_BY]->(s)
    # """
    q = """
    MATCH (a:CodePiece), (b:PrivacyPolicySection)
    WHERE a.is_requirement = true AND a.id = $req_id AND b.id = $section_id
    CREATE (a)-[r:IS_FULFILLED_BY]->(b)
    RETURN r
    """
    result = graph.query(q, params={"req_id": req_id, "section_id": section_id})
    if not result:
        logger.warn(
            f"Could not create IS_FULFILLED_BY relationship for {req_id} -> {section_id}"
        )


def mark_not_fulfilled_by(req_id, section_id):
    q = """
    MATCH (a:CodePiece), (b:PrivacyPolicySection)
    WHERE a.is_requirement = true AND a.id = $req_id AND b.id = $section_id
    CREATE (a)-[r:IS_NOT_FULFILLED_BY]->(b)
    RETURN r
    """
    result = graph.query(q, params={"req_id": req_id, "section_id": section_id})
    if not result:
        logger.warn(
            f"Could not create IS_NOT_FULFILLED_BY relationship for {req_id} -> {section_id}"
        )


# def codepiece_requirement_similarity_search(
#     vector: list,
#     embedding_name="gte-large",
#     ccpa_only=True,
#     limit=10,
# ):
#     q = f"""
#     CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
#     YIELD node, score
#     WHERE node.is_requirement = true
#     {"AND node.law_code = 'CIV' and node.division = '3.' and node.part = '4.' and node.title = '1.81.5.'" if ccpa_only else ""}
#     RETURN node, score
#     """
#     results = graph.query(
#         q,
#         params={
#             "index_name": embedding_index_name(embedding_name),
#             "embedding": vector,
#             "limit": limit,
#         },
#     )
#     return list(sorted(results, key=lambda r: r["score"], reverse=True))


def create_codepiece_requirements_index():
    q = """
    CREATE INDEX index_name FOR (n:CodePiece) ON (n.law_code, n.division, n.part, n.title, n.is_requirement)
    """
    graph.query(q)

"""Deeper 2022, All Rights Reserved
"""
import logging

from db.neo4j.connector import Neo4jDBConnector
from tasks.nlp import openai_text_extraction
from tasks.graph import add_describers_nodes


async def analyze_deep_request(
    neo4j_connector: Neo4jDBConnector,
    deep_request: str,
    node_id: int
) -> None:
    """Chained tasks to perform a full analasyis of a deep request
    
    :param Neo4jDBConnector neo4j_connector:
    :param str deep_request:
    :param int node_id:
    """
    logging.info(f'About to analyse deep request ({node_id}): {deep_request}')
    data = await openai_text_extraction(deep_request)
    with neo4j_connector.use_session() as neo4j_session:
        await add_describers_nodes(neo4j_session, data, node_id)
    logging.info(f'Analysing deep request ({node_id}) is done')

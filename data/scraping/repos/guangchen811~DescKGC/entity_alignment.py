import argparse

from langchain.chat_models import ChatOpenAI

from DescKGC.procedures.align_across_subgraphs.base import (
    align_source_and_candidates_with_chain,
    get_entity_type_uuids,
    query_from_specific_type_uuids,
    select_candidate_entities_uuids,
    merge_entities_with_chain,
)
from DescKGC.procedures.build_config import load_config
from DescKGC.tools.align.base import init_align_chain, init_entity_merge_chain
from DescKGC.tools.align.parser import EntityAlignOutputParser, EntityMergeOutputParser
from DescKGC.tools.db_manager import DBManager
from DescKGC.procedures.align_across_subgraphs.utils import add_one_merged_entity


def add_arguments(parser):
    parser.add_argument(
        "--entity_types",
        type=list,
    )


def main(args):
    config = load_config()
    if args.entity_types is None:
        entity_types = config["entity_alignment"]["entity_types"]
    else:
        entity_types = args.entity_types
    shortenings = config["shortenings"]
    metadata_keys = config["extractor"]["entity"]["vs_key_info"]["metadata_keys"]
    embedding_key = config["extractor"]["entity"]["vs_key_info"]["embedding_key"]

    db_manager = DBManager(**config["neo4jdb"], **config["chromadb"])
    llm = ChatOpenAI(temperature=config["llm"]["temperature"])
    topic = config["topic"]
    align_chain = init_align_chain(llm=llm)
    align_parser = EntityAlignOutputParser()
    merge_chain = init_entity_merge_chain(llm=llm)
    merge_parser = EntityMergeOutputParser()
    entity_type_uuids_dict = get_entity_type_uuids(db_manager, entity_types)

    for entity_type, uuids in entity_type_uuids_dict.items():
        for uuid in uuids:
            try:
                similar_entities = query_from_specific_type_uuids(db_manager, [entity_type], uuid)
                candidate_uuids = select_candidate_entities_uuids(
                    threshold=config["entity_alignment"]["threshold"],
                    similar_entities=similar_entities,
                    src_entity_uuid=uuid,
                )
                if len(candidate_uuids) > 0:
                    selected_entities = align_source_and_candidates_with_chain(
                        topic, db_manager, align_chain, align_parser, uuid, candidate_uuids
                    )
                    selected_entities_ids = [entity[0] - 1 for entity in selected_entities]
                    selected_candidate_uuids = [candidate_uuids[i] for i in selected_entities_ids]
                    new_entity = merge_entities_with_chain(
                        db_manager, topic, merge_chain, merge_parser, uuid, selected_candidate_uuids
                    )
                    new_entity["type"] = entity_type
                    entity_shortning = shortenings[entity_type]
                    uuids_to_link = [uuid] + selected_candidate_uuids
                    uuid_ = add_one_merged_entity(
                        db_manager=db_manager,
                        id_type="uuid",
                        id_values=uuids_to_link,
                        entity=new_entity,
                        shortning=entity_shortning,
                    )
                    metadata = {metadata_key: new_entity[metadata_key] for metadata_key in metadata_keys}
                    metadata["embedding_source"] = "description"
                    metadata["doc_source_type"] = "generated"
                    db_manager.vector_db.add(
                        documents=[new_entity[embedding_key]],
                        metadatas=[metadata],
                        ids=[uuid_],
                    )
                else:
                    print("No similar entities found for entity type: ", entity_type)
            except Exception as e:
                print(f"Error when aligning entities from {uuid}: {e}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)

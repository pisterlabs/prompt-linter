from langchain.vectorstores.faiss import FAISS
from pathlib import Path
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import logging
import pandas as pd


logger = logging.getLogger(__name__)


"""Parse field from JSON or raise error if missing"""


def parse_field(request_json, field: str):
    if request_json and field in request_json:
        return request_json[field]
    else:
        raise ValueError(f"JSON is invalid, or missing a '${field}' property")


def get_dbs():
    dir = Path(__file__).parent.absolute()
    general_embeddings, in_depth_embeddings = create_embeddings()

    general_faiss_index_path = dir.joinpath("cache/faiss_index_general")
    in_depth_faiss_index_path = dir.joinpath("cache/faiss_index_in_depth")
    voting_roll_df_path = dir.joinpath("cache/parsed_voting_rolls.csv")
    db_general = FAISS.load_local(general_faiss_index_path, general_embeddings)
    db_in_depth = FAISS.load_local(in_depth_faiss_index_path, in_depth_embeddings)
    logger.info("Loaded databases from faiss_index_general and faiss_index_in_depth")
    voting_roll_df = pd.read_csv(voting_roll_df_path)
    return db_general, db_in_depth, voting_roll_df


def create_embeddings():
    llm = OpenAI()

    general_prompt_template = """
    As an AI assistant tasked with generating brief general summaries, your role is to provide succinct, balanced information from the transcripts of New Orleans City Council meetings in response to the question "{question}". The response should not exceed one paragraph in length. If the available information from the transcripts is insufficient to accurately summarize the issue, please respond with 'Insufficient information available.' If the question extends beyond the scope of information contained in the transcripts, state 'I don't know.'
    Answer:"""

    in_depth_prompt_template = """
    As an AI assistant tasked with providing in-depth dialogical summaries, your role is to provide comprehensive information from the transcripts of New Orleans City Council meetings. Your response should mimic the structure of a real conversation, often involving more than two exchanges between the parties. The dialogue should recreate the actual exchanges that occurred between city council members and external stakeholders in response to the question "{question}". For specific queries related to any votes that took place, your response should include detailed information. This should cover the ordinance number, who moved and seconded the motion, how each council member voted, and the final outcome of the vote. For each statement, response, and voting action, provide a summary, followed by a direct quote from the meeting transcript to ensure the context and substance of the discussion is preserved. If a question is about the voting results on a particular initiative, include in your response how each council member voted, if they were present, and if there were any abstentions or recusals. Always refer back to the original transcript to ensure accuracy. If the available information from the transcripts is insufficient to accurately answer the question or recreate the dialogue, please respond with 'Insufficient information available.' If the question extends beyond the scope of information contained in the transcripts, state 'I don't know.'
    Answer:"""

    general_prompt = PromptTemplate(
        input_variables=["question"], template=general_prompt_template
    )
    in_depth_prompt = PromptTemplate(
        input_variables=["question"], template=in_depth_prompt_template
    )

    llm_chain_general = LLMChain(llm=llm, prompt=general_prompt)
    llm_chain_in_depth = LLMChain(llm=llm, prompt=in_depth_prompt)

    base_embeddings = OpenAIEmbeddings()

    general_embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain_general, base_embeddings=base_embeddings
    )
    in_depth_embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain_in_depth, base_embeddings=base_embeddings
    )

    return general_embeddings, in_depth_embeddings


def sort_retrived_documents(doc_list):
    docs = sorted(doc_list, key=lambda x: x[1], reverse=True)

    third = len(docs) // 3

    highest_third = docs[:third]
    middle_third = docs[third : 2 * third]
    lowest_third = docs[2 * third :]

    highest_third = sorted(highest_third, key=lambda x: x[1], reverse=True)
    middle_third = sorted(middle_third, key=lambda x: x[1], reverse=True)
    lowest_third = sorted(lowest_third, key=lambda x: x[1], reverse=True)

    docs = highest_third + lowest_third + middle_third
    return docs

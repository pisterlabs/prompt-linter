if True:
    import sys
    sys.path.append("../")

import langchain
from langchain.chains import StuffDocumentsChain, LLMChain, ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
import asyncio
import json


# local imports
from src.utils.logger import Logger
# set logger
logger = Logger(__name__).get_logger()

class SuggestionGeneratorException(Exception):
    """Custom class for handling Suggestion Generation Exceptions"""


class SuggestionGenerator:

    def __init__(self, llm, type_llm: str, debug: bool = False) -> None:
        """Suggestion generator"""
        try:
            self.debug = debug
            self.llm = llm
            self.type_llm = type_llm

            # get prompts
            self.PROMPT_DOC, self.PROMPT_MAP_SUGGESTION, self.PROMPT_REDUCE_SUGGESTION = self.get_prompts()

            # Instance splitter
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=0.1
            )

            # build map reduce chain
            self.map_reduce_chain = self.build_map_reduce_suugestion_chain()
        except Exception as e:
            logger.error(f"Error in SuggestionGenerator: {e}")
            raise e

    async def run(self, text, return_dict: bool = True):
        """Run the suggestion generator"""
        # split in chunks
        try:
            chunks = self.splitter.split_text(text)

            # transform all chunks (string) to Document type
            chunks_docs = [langchain.schema.document.Document(
                page_content=chunk) for chunk in chunks]

            # run map reduce in parrallel
            raw_output = await asyncio.gather(
                self.map_reduce_chain.arun(chunks_docs))

            # Parse output
            if type(raw_output) == list:
                raw_output = raw_output[0]

            # format output
            if return_dict:
                try:
                    return str(self.parse_output_suggestions(raw_output))
                except:
                    return str(raw_output)

            # transform output to string
        except Exception as e:
            logger.error(f"Error in SuggestionGenerator: {e}")
            raise e

        return raw_output

    def run_sync(self, text, return_dict: bool = True):
        """Run the suggestion generator"""
        # split in chunks
        chunks = self.splitter.split_text(text)

        # transform all chunks (string) to Document type
        chunks_docs = [langchain.schema.document.Document(
            page_content=chunk) for chunk in chunks]

        # run map reduce in parrallel
        raw_output = self.map_reduce_chain.run(chunks_docs)

        # Parse output
        if type(raw_output) == list:
            raw_output = raw_output[0]

        # format output
        if return_dict:
            try:
                return str(self.parse_output_suggestions(raw_output))
            except:
                return str(raw_output)

        # transform output to string
        return raw_output

    def get_prompts(self):
        """Return prompts"""
        if self.type_llm == "bedrock":
            from src.prompts.prompts_template_bedrock import PROMPT_DOC, PROMPT_MAP_SUGGESTION, PROMPT_REDUCE_SUGGESTION
        else:
            from src.prompts.prompts_template import PROMPT_DOC, PROMPT_MAP_SUGGESTION, PROMPT_REDUCE_SUGGESTION
        return PROMPT_DOC, PROMPT_MAP_SUGGESTION, PROMPT_REDUCE_SUGGESTION

    def build_map_reduce_suugestion_chain(self):
        # create map chain
        map_chain = LLMChain(
            llm=self.llm,
            prompt=self.PROMPT_MAP_SUGGESTION,
            verbose=self.debug)

        # create reduce chain
        reduce_chain = LLMChain(
            llm=self.llm,
            prompt=self.PROMPT_REDUCE_SUGGESTION,
            verbose=self.debug
        )

        # transform list of docs to string
        docs_to_string_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_prompt=self.PROMPT_DOC,
            document_variable_name='context')

        # Combine docs by recursivelly reducing them
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=docs_to_string_chain,
        )

        # map reduce chain
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
        )

        return map_reduce_chain

    @staticmethod
    def parse_output_suggestions(response_str: str) -> list:
        """Parse output suggestions
        Args:
            response: response from the chain
        Returns:
            List of tuples with the original parragraph and the modified parragraph
            example:
            [
                ("original parragraph 1", "modified parragraph 1"),
                ("original parragraph 2", "modified parragraph 2"),
                ("original parragraph 3", "modified parragraph 3"),
            ]

        """
        response_dict = json.loads(response_str)

        original_parragraph_ls = []
        for r in response_dict['original_parragraph']:
            if type(r) == str:
                original_parragraph_ls.append(r)
            if type(r) == dict:
                if 'Original parragraph' in list(r.keys()):
                    original_parragraph_ls.append(r['Original parragraph'])
                if 'Original Parragraph' in list(r.keys()):
                    original_parragraph_ls.append(r['Original Parragraph'])

        modified_parragraph_ls = []
        for r in response_dict['modified_parragraph']:
            if type(r) == str:
                modified_parragraph_ls.append(r)
            if type(r) == dict:
                if 'Modified parragraph' in list(r.keys()):
                    modified_parragraph_ls.append(r['Modified parragraph'])
                if 'Modified Parragraph' in list(r.keys()):
                    modified_parragraph_ls.append(r['Modified Parragraph'])

        explanation_ls = []
        for r in response_dict['explanation']:
            if type(r) == str:
                explanation_ls.append(r)
            if type(r) == dict:
                if 'Explanation' in list(r.keys()):
                    explanation_ls.append(r['Explanation'])
                if 'explanation' in list(r.keys()):
                    explanation_ls.append(r['explanation'])

        # join the original parragraphs and modified parragraphs as tuple
        res_parsed = {}
        res_parsed['suggestions'] = []
        for i in range(len(original_parragraph_ls)):
            res_parsed['suggestions'].append({
                'orgiginal_parragraph': original_parragraph_ls[i],
                'modified_parragraph': modified_parragraph_ls[i],
                'explanation': explanation_ls[i],
            })

        # res_parsed = list(zip(original_parragraph_ls,
            #   modified_parragraph_ls, explanation_ls))
        return res_parsed


if __name__ == "__main__":
    from langchain.document_loaders import PDFMinerLoader
    from src.base_llm import BaseLLM

    # Args
    path_file = "../docs_example/contract.pdf"
    debug = True
    # read document
    doc = PDFMinerLoader(path_file).load()[0].page_content

    llm_base = BaseLLM(debug=debug)
    # Instance
    suggestion = SuggestionGenerator(
        llm=llm_base.llm,
        type_llm=llm_base.type_llm,
        debug=debug)

    # Test map chain

    # run chain
    result = suggestion.run_sync(text=doc[:1000], return_dict=False)

    # It's not possible run as in jupyer
    # result = asyncio.run(suggestion.run(text=doc[:1000]))
    print(result)

import fitz
import pandas as pd
import re
import os
import logging
from langchain.docstore.document import Document
from langchain.utilities.arxiv import ArxivAPIWrapper
from typing import List, Generator
from arxiv.arxiv import Result



logger = logging.getLogger(__name__)


class ArxivPDF(ArxivAPIWrapper):

    def load(self, query, parse_pdf=True, split_sections=False, keep_pdf=False):
        """
        This overrides the load method in ArxivAPIWrapper to keep the downloaded PDF
        Run Arxiv search and get the article texts plus the article meta information.
        See https://lukasschwab.me/arxiv.py/index.html#Search

        Returns: a list of documents with the document.page_content in text format

        """
        try:
            import fitz
        except ImportError:
            raise ImportError(
                "PyMuPDF package not found, please install it with "
                "`pip install pymupdf`"
            )

        try:
            results: Generator[Result] = self.arxiv_search(  # type: ignore
                query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.load_max_docs
            ).results()
        except self.arxiv_exceptions as ex:
            logger.debug("Error on arxiv: %s", ex)
            return []

        docs: List[Document] = []
        for i, result in enumerate(results):
            if i >= 1:
                raise NotImplementedError("Only one result is supported for now")

            doc_file_name = result._get_default_filename()
            if not os.path.exists(doc_file_name):
                logger.info(f"Downloading {doc_file_name}...")
                result.download_pdf()

            try:
                with fitz.open(doc_file_name) as doc_file:
                    text: str = "".join(page.get_text() for page in doc_file)
            except FileNotFoundError as f_ex:
                logger.debug(f_ex)
                continue

            if self.load_all_available_meta:
                extra_metadata = {
                    "entry_id": result.entry_id,
                    "published_first_time": str(result.published.date()),
                    "comment": result.comment,
                    "journal_ref": result.journal_ref,
                    "doi": result.doi,
                    "primary_category": result.primary_category,
                    "categories": result.categories,
                    "links": [link.href for link in result.links],
                }
            else:
                extra_metadata = {}
            metadata = {
                "Published": str(result.updated.date()),
                "Title": result.title,
                "Authors": ", ".join(a.name for a in result.authors),
                "Summary": result.summary,
                **extra_metadata,
            }
            doc = Document(
                page_content=text[: self.doc_content_chars_max], metadata=metadata
            )
            docs.append(doc)
            # this is the only change from the original method
            if parse_pdf:
                pdf_docs = self.split_text(doc_file_name, split_sections=split_sections)
            if not keep_pdf:
                os.remove(doc_file_name)
        return docs, pdf_docs, doc_file_name

    @staticmethod
    def _is_footnote(row: pd.Series) -> bool:
        return re.match(r'^\d+[a-zA-z]+', row.text.strip()) is not None

    @staticmethod
    def _is_section_header(row: pd.Series) -> bool:
        return re.match(r'^\d+\. ', row.text.strip()) is not None
    
    @staticmethod
    def is_two_column(blocks: pd.DataFrame, page_width: int, tolerance: float = 0.2) -> bool:
        """
        Check if the document is in two column format
        Args:
            blocks: a dataframe with the text blocks from the pdf
        """
        # Get the median x centroid of each block and determine if it's close to the centre
        # of the page
        x_centroid = (blocks.x0 + blocks.x1) / 2
        centre = page_width // 2
        one_column = abs(x_centroid.median() - centre) < tolerance * page_width
        return not one_column
    
    @staticmethod
    def parse_two_column(blocks: pd.DataFrame, page_width: int, page_height: int) -> pd.DataFrame:
        """
        Parse a two column document
        """
        pass

        
    def get_text_dataframe(self, fname) -> pd.DataFrame:

        with fitz.open(fname) as doc:
            # get the width and height of the first page
            dfs = []
            for i, page in enumerate(doc):
                width, height = page.rect[2:]
                centre = width // 2
                pdata = page.get_text("blocks")
                df = pd.DataFrame(pdata, columns=['x0', 'y0', 'x1', 'y1', 'text', 'block_no', 'block_type'])
                # assume that text to the left of center are in the first column
                # assume that text to the right of center are in the second column
                # try to extract the title and author list from the first page
                # split left and right columns
                # ignore blocks that span both columns
                if self.is_two_column(df, width):
                    logger.debug(f"Got two column document for page {i}")
                    df_left =  df.loc[(df.x0 < centre) & (df.x1 < centre)]
                    df_right =  df.loc[(df.x0 > centre) & (df.x1 > centre)]

                    if i == 0:
                        # Assume the title block is the first one that spans the centre column
                        title_block = df.loc[(df.x0 < centre) & (df.x1 > centre) & (df.y0 < 0.2 * height)]
                        # add title block to left column
                        df_left = pd.concat([title_block, df_left])

                    df_out = pd.concat([df_left, df_right])
                else:
                    logger.debug(f"Got one column document for page {i}")
                    # parse one column format
                    df_out = df.copy()

                # filter out images
                df_out = df_out.loc[df_out.block_type == 0]
                # filter out vertical text
                df_out = df_out.loc[df_out.x1 - df_out.x0 > 0.5 * (df_out.y1 - df_out.y0)]
                # filter out footnotes
                try:
                    df_out = df_out.loc[~df_out.apply(self._is_footnote, axis=1)]
                except:
                    import pdb; pdb.set_trace()
                    pass
                df_out['page_no'] = i

                dfs.append(df_out)

        return pd.concat(dfs)

    def split_text(self, fname: str, split_sections:bool=False) -> List[Document]:
        """ Extract text from a an arxiv pdf in 2 column format
        Args:
            fname: the filename of the pdf
            split_sections: if True, split the text into sections, otherwise split into pages
        """

        df = self.get_text_dataframe(fname)
        sections = [""]
        section_names = ["None"]
        prev_page = -1
        for ind, row in df.iterrows():
            if split_sections:
                if self._is_section_header(row):
                    sections.append("")
                    section_names.append(row.text.strip())
            else:
                if row.page_no != prev_page:
                    sections.append("")

            sections[-1] += row.text + "\n"
            prev_page = row.page_no
        
        if split_sections:
            return [Document(page_content=section, metadata={"section_name": section_name}) for section, section_name in zip(sections, section_names)]
        else:
            return [Document(page_content=section) for section in sections]



if __name__ == "__main__":
    fname = "2302.00923v4_clean.pdf"
    pdf = ArxivPDF(fname)
    print(f"Extracting text from {fname}")
    docs = pdf.split_text(True)

    outfname = fname.replace('pdf', 'txt')
    print(f"Writing to {outfname}")
    with open(outfname, 'w') as f:
        f.write("\n\n".join([page.page_content for page in docs]))
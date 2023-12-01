# #pip install beautifulsoup4
#
# #wget -r -A.html -P scraped_docs_data https://freemocap.readthedocs.io/en/latest/
#
# from langchain.document_loaders import ReadTheDocsLoader
#
#
#
# import asyncio
# from pathlib import Path
# from typing import Union
#
# from dotenv import load_dotenv
# from langchain.chat_models import ChatOpenAI, ChatAnthropic
# from langchain.prompts import ChatPromptTemplate
# from magic_tree.controller.builders.directory_tree_builder import DirectoryTreeBuilder
#
# load_dotenv()
#
# SUMMARY_PROMPT = """
# Summarize the following text, which is a section of an NIH R01.
#
# The text is writtne in `.tex`. In your summary, you should roughly match the
# structure defined in the `tex` file using markdown heading levels to match tex
# `section` definitions (i.e. # - `section`, ## -`subsection` etc)
#
#  So your summary should look something like this:
#  FILE_PATH - [e.g. /document/specific_aims/main_specific_aims.tex]
#  STATUS - [an estimate of how close to "done" this is, in a few words]
#  ## [section name]
#     ### [subsection name]
#         #### [subsubsection name]
#  ## [another section name]
#  ## Notes:
#     - Strengths
#         - [list of strengths]
#     - Weaknesses
#         - [list of weaknesses]
#
#
#  DOCUMENT DRAFT TEXT:
#
#  {text}
#
# """
#
#
# def create_component_summary_chain():
#     prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
#     model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
#     chain = prompt | model
#     return chain
#
#
# GLOBAL_SUMMARY_PROMPT = """
# The following text is an early, incomplete draft of an NIH R01 Proposal
#
# Summarize the main points of the proposal, including the following sections:
# # Specific Aims
# # Research Strategy
# ## Significance
# ## Innovation
# ## Approach
# ### Aim 1
# ### Aim 2
#
# Then list the Strengths and Weaknesses of the proposal
#
#  CURRENT PROPOSAL TEXT:
#
# {text}
# """
#
#
# def create_global_summary_chain():
#     prompt = ChatPromptTemplate.from_template(GLOBAL_SUMMARY_PROMPT)
#
#     model = ChatAnthropic(temperature=0)
#     chain = prompt | model
#     return chain
#
#
# async def auto_freemocap_documentation(document_root_path: Union[Path, str]):
#     loader = ReadTheDocsLoader(document_root_path, features="html.parser")
#     docs = loader.load()
#
#     output_text = ""
#
#     component_summary_chain = create_component_summary_chain()
#     global_summary_chain = create_global_summary_chain()
#     input_texts = []
#     all_text = []
#     for doc in docs:
#         file_content = document_root_path.read_text()
#
#         text_to_analyze = (f"{str(document_root_path)}\n"
#                            f"{file_content}")
#         all_text.append(text_to_analyze)
#         input_texts.append({"text": text_to_analyze})
#
#     all_summaries = await component_summary_chain.abatch(input_texts)
#     for summary in all_summaries:
#         file_summary = (f"+++++++++++++++++++++++++++++++++++\n\n"
#                         f"{summary.content}\n\n"
#                         f"-----------------------------------\n\n")
#         print(file_summary)
#         output_text += file_summary
#
#     print(output_text)
#
#     global_summary = global_summary_chain.invoke({"text": "\n".join(all_text)}).content
#
#     output_text += (f"=============================================================\n\n"
#                     f"=============================================================\n\n"
#                     f"___ \n > Global Summary \n {global_summary}\n\n")
#
#     document_tree = DirectoryTreeBuilder.from_path(path=document_root_path)
#
#     document_tree_string = f"```\n\n{document_tree.print()}\n\n```\n\n"
#     print(document_tree_string)
#     output_text += document_tree_string
#
#     with open("document_summary-diataxis0.md", "w", encoding="utf-8") as file:
#         file.write(output_text)
#
#
# if __name__ == "__main__":
#     # document_root_path_in = Path("./diataxis-documentation-framework-repo")
#     document_root_path_in = Path(
#         "/Users/jon/github_repos/jonmatthis/pauljon/scraped_docs_data/freemocap.readthedocs.io/en/latest")
#     asyncio.run(auto_freemocap_documentation(document_root_path=document_root_path_in))

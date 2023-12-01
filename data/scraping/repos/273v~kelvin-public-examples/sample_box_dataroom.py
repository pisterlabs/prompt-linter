"""
Show an example using Kelvin Source, Kelvin NLP, and Kelvin API to:
1. Connect to a Box share folder
2. Send all files to Kelvin Document Index API to extract text
3. Summarize the text using an LLM
4. Answer a question about the document using an LLM
"""

# imports
import asyncio
import textwrap

# project imports
from kelvin.api.document_index.async_client import KelvinDocumentIndexAsyncClient
from kelvin.nlp.llm.engines.openai_engine import OpenAIEngine
from kelvin.nlp.llm.summarize.recursive_split_summarizer import RecursiveSplitSummarizer
from kelvin.nlp.llm.summarize.text_memoizer import TextMemoizer
from kelvin.source.box_source import BoxSource, DEFAULT_CREDENTIAL_PATH, DEFAULT_TOKEN_PATH

async def main():
    # setup a kelvin doc index client
    doc_index_client = KelvinDocumentIndexAsyncClient()

    # setup an LLM engine
    llm35_engine = OpenAIEngine(model="gpt-3.5-turbo")
    llm4_engine = OpenAIEngine(model="gpt-4")
    summarizer = RecursiveSplitSummarizer(llm35_engine)
    memoizer = TextMemoizer(engine=llm4_engine)

    # setup box source
    folder = BoxSource("https://acmeinc.app.box.com/folder/123412341234", recursive=True)

    summaries = {}
    for file in folder:
        # send to kelvin doc index for text content and vectors
        doc_object = await doc_index_client.upload_document(file_object=file.to_filelike(), file_name=file.name)
        if doc_object is None:
            continue
        doc_content = await doc_index_client.get_document_contents(document_id=doc_object["id"], content_type="text/plain")
        if doc_content is None:
            continue

        # summarize each document
        for doc in doc_content:
            summaries[file.name] = summarizer.get_summary(doc["content"].decode(), progress=True, final_summarize=False)
            print(f"Summary for {file.name}: {textwrap.fill(summaries[file.name], 80)}" + "\n" + "=" * 80 + "\n")

    # write a memo about it all
    combined_summary = "\n".join(f"File: {key}\nSummary: {value}\n" for key, value in summaries.items())
    memo = memoizer.get_summary(combined_summary, progress=False)
    print(memo)


if __name__ == "__main__":
    asyncio.run(main())

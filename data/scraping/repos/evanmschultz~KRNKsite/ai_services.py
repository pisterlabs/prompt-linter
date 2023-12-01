import asyncio
import json
from enum import Enum
from services.utils import remove_after_references
from enum import Enum
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatAnthropic
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import OnlinePDFLoader


class SummaryLength(Enum):
    FULL = "full"
    SHORT = "short"


full_summary_prompt_template = """You are an expert in the field of {topic} and you are writing a summary of the paper {title} working for 
a {topic} blog:


{paper_content}


Write a detailed 100-150 sentence summary article of the paper in a compelling blog post, include all majors points and conclusions. You 
want readers to understand the key takeaways without having to read the full paper.

DETAILED SUMMARY:"""

FULL_SUMMARY_PROMPT = PromptTemplate(
    template=full_summary_prompt_template,
    input_variables=["topic", "title", "paper_content"],
)


short_summary_prompt_template = """You are an expert in the field of {topic} and you are writing a summary of the paper {title} working for 
a {topic} blog:


{paper_content}

Write a short summary article of the paper in less than 20 sentences in a compelling blog post, include only majors points and conclusions. You 
want readers to understand the key takeaways to be encouraged to read the full paper.

SHORT SUMMARY:"""

SHORT_SUMMARY_PROMPT = PromptTemplate(
    template=short_summary_prompt_template,
    input_variables=["topic", "title", "paper_content"],
)


async def summarize_pdf(
    pdf_url: str,
    summary_length: SummaryLength = SummaryLength.FULL,
    title: str = "",
    topic: str = "",
) -> str | None:
    """
    Summarize a PDF using Anthropic's Claude 2.

    Produces two summaries: a short summary and a long summary.

    Args:
        pdf_path (str): The path to the PDF to summarize.

    Returns:
        dict: The summaries of the PDF.
            {
                "short_summary": str,
                "long_summary": str,
            }
    """
    loader = OnlinePDFLoader(pdf_url)
    data = loader.load()

    full_paper_content: str = data[0].page_content
    paper_content: str = remove_after_references(full_paper_content)
    print(f"""\n{'_'*80}\nfull_paper_length:\n\n{len(full_paper_content)}\n{'_'*80}""")
    print(f"""\n{'_'*80}\npaper_length:\n\n{len(paper_content)}\n{'_'*80}""")

    input_list: list[dict] = [
        {
            "topic": topic,
            "title": title,
            "paper_content": paper_content,
        }
    ]

    if summary_length == SummaryLength.FULL:
        chosen_prompt = FULL_SUMMARY_PROMPT
    else:
        chosen_prompt = SHORT_SUMMARY_PROMPT

    llm = ChatAnthropic(
        max_tokens=5000,
        temperature=0.1,
        # streaming=False,  # Set to True to stream the output
        # callbacks=[
        #     StreamingStdOutCallbackHandler()
        # ],  # Callbacks to handle the streaming output
    )

    summary_chain = LLMChain(llm=llm, prompt=chosen_prompt)
    try:
        summary_result: list[dict[str, str]] = await summary_chain.aapply(input_list)
        summary: str = (
            json.loads(json.dumps(summary_result))[0]["text"].split(":\n\n")[1].strip()
        )
        print(f"""\n{'_'*80}\n{summary_length}\n{summary}\n{'_'*80}""")
        return summary

    except Exception as e:
        print(f"""\n{'_'*80}\nException:\n\n{e}\n{'_'*80}""")
        return


# Testing the function
async def main() -> None:
    task1 = summarize_pdf(
        "https://arxiv.org/pdf/2106.01548.pdf",
        SummaryLength.FULL,
        topic="AI > Computer Vision",
        title="WHEN VISION TRANSFORMERS OUTPERFORM RESNETS WITHOUT PRE-TRAINING OR STRONG DATA AUGMENTATIONS",
    )
    task2 = summarize_pdf(
        "https://arxiv.org/pdf/2106.01548.pdf",
        SummaryLength.SHORT,
        topic="AI > Computer Vision",
        title="WHEN VISION TRANSFORMERS OUTPERFORM RESNETS WITHOUT PRE-TRAINING OR STRONG DATA AUGMENTATIONS",
    )

    await asyncio.gather(task1, task2)


if __name__ == "__main__":
    asyncio.run(main())

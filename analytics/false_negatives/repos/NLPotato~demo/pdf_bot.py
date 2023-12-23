from modules.base import BaseBot
from modules.templates import (
    PDF_PREPROCESS_TEMPLATE,
    PDF_PREPROCESS_TEMPLATE_WITH_CONTEXT,
)
from modules.preprocessors import PDFPreprocessor

from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import MarkdownTextSplitter


PATH = "data/test/[제안요청서]AIBanker서비스구축을위한데이터셋구축(우리은행)_F.pdf"
pdf_loader = PDFPlumberLoader(
    PATH,
)
preprocessor = PDFPreprocessor(
    prompt=PDF_PREPROCESS_TEMPLATE_WITH_CONTEXT,
)

bot = BaseBot.from_new_collection(
    loader=pdf_loader,
    collection_name="woori_pdf_prev_md",
    preprocessor=preprocessor,
)

# questions = [
#     "제안 평가 중 기술 및 업무 부문에 대한 평가에 대한 평가는 어떻게 돼?",
#     "이번 사업 개요는?",
#     "작업 결과 평가 기준은?",
#     "제안서 목차는s?",
# ]
# for q in questions:
#     resp = bot(q)
#     print(resp["answer"])

import logging

from text.export_processors import TextExportProcessor
from text.summarizers.openai_adapter import OpenAIProcessor, PromtProcessor
from text.zoom.processor import ZoomTranscriptionProcessor

logging.basicConfig(level=logging.INFO)

# zoom adapter -> promts -> openai adapter -> responses -> texts -> post_transformers
if __name__ == '__main__':
    t = "/Users/arrtz3/code/home/summarization/artifacts/GMT20220811-171211_Recording.transcript.vtt"
    promt_processor = PromtProcessor(2, 400, 4001)
    openai_processor = OpenAIProcessor()
    txt_processor = TextExportProcessor()
    pipeline = [
        ZoomTranscriptionProcessor(),
        promt_processor, openai_processor,
        promt_processor,
        openai_processor,
        txt_processor
    ]
    args = t
    for step in pipeline:
        args = step(args)
    print(args)
    # text_processor = TextProcessor(ZoomTranscriptionProcessor())
    # tldr = text_processor.process(t)
    # print(tldr)

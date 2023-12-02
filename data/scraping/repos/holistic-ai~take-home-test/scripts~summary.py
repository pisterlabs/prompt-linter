from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain.document_loaders import UnstructuredFileLoader

tokenizer = AutoTokenizer.from_pretrained("pszemraj/pegasus-x-large-book-summary")
model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/pegasus-x-large-book-summary")


summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
)


loader = UnstructuredFileLoader("InvoiceXpert.pdf")

docs = loader.load()

params = {
    "max_length": 256,
    "min_length": 8,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "repetition_penalty": 3.5,
    "length_penalty": 0.2,
    "encoder_no_repeat_ngram_size": 3,
    "num_beams": 4,
}


result = summarizer(docs[0].page_content, **params)
print(f"summarized inputs are:\n\n{result[0]['summary_text']}")

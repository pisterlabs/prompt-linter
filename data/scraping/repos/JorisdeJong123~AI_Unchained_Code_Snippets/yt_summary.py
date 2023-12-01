from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter

# Load Transcript
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=5sLYAQS9sWQ", language=["en", "en-US"])
transcript = loader.load()

# Split Transcript
splitter = TokenTextSplitter(model_name="gpt-3.5-turbo-16k", chunk_size=10000, chunk_overlap=100)
chunks = splitter.split_documents(transcript)

# Set up LLM
openai_api_key = "YOUR_OPENAI_API_KEY"
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k", temperature=0.3)

# Summarize
summarize_chain = load_summarize_chain(llm=llm, chain_type="refine", verbose=True)
summary = summarize_chain.run(chunks)

# Write summary to file
with open("summary.txt", "w") as f:
    f.write(summary)
# https://api.python.langchain.com/en/latest/callbacks/langchain.callbacks.file.FileCallbackHandler.html#langchain.callbacks.file.FileCallbackHandler
from loguru import logger

from langchain.callbacks import FileCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# logfile = "./output.log"

# logger.add(logfile, colorize=True, enqueue=True)
# handler = FileCallbackHandler(logfile)

# model = ChatOpenAI()
# prompt = ChatPromptTemplate.from_template("1 + {number} = ")

# # this chain will both print to stdout (because verbose=True) and write to 'output.log'
# # if verbose=False, the FileCallbackHandler will still write to 'output.log'
# chain = LLMChain(
#     llm=model,
#     prompt=prompt,
#     callbacks=[handler],
#     verbose=True,
# )

# answer = chain.run(number=2)

# logger.info(answer)

from ansi2html import Ansi2HTMLConverter

with open("output.log", "r") as f:
    content = f.read()

conv = Ansi2HTMLConverter()
html = conv.convert(content, full=True)

print(html)

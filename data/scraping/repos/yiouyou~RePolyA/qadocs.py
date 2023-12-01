from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from repolya.paper._paperqa import Docs

from repolya._const import PAPER_PDF
from repolya._log import logger_paper


# llm_stream = ChatOpenAI(model='gpt-4', callbacks=[StreamingStdOutCallbackHandler()], streaming=True)
llm = ChatOpenAI(model='gpt-4')
llm_summary = ChatOpenAI(model='gpt-3.5-turbo')
docs = Docs(llm=llm, summary_llm=llm_summary)


@logger_paper.catch
def qadocs(_query, _pathlist):
    logger_paper.debug(f"query: '{_query}'")
    for _file in _pathlist:
        try:
            if _file.lower().endswith('.pdf') or _file.lower().endswith('.txt'):
                docs.add(_file, chunk_chars=500)
                # print(_file)
                logger_paper.debug(_file)
        except ValueError as e:
            # print('Could not read', _file, e)
            logger_paper.debug('Could not read', _file, e)
    _ans = docs.query(_query)
    logger_paper.debug(f"question: '{_ans.question}'")
    logger_paper.debug(f"answer: '{_ans.answer}'")
    logger_paper.debug(f"references: '{_ans.references}'")
    logger_paper.debug(f"context: '{_ans.context}'")
    ##### formatted_answer, question/answer/references, context
    return _ans


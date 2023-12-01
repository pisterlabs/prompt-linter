import wikipedia as wiki

from lib.TreeTypes import UnanswerableQuestion

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

from query.wiki_queries import chain as chain_wiki_queries
from query.answer_atomic import chain as chain_answer_atomic

embeddings = OpenAIEmbeddings()




class WikipediaAnswerer:
    def __init__(self, llm, logger):
        self.llm=llm
        self.wiki_queries = chain_wiki_queries(llm)
        self.answer_atomic = chain_answer_atomic(llm)
        self.log = logger

    def page2doc(self, page, type='summary'):
        self.log(type, page)

        try:
            return Document(
                page_content = wiki.summary(page, auto_suggest=False) if type=='summary' else
                            wiki.page(title=page).content if type=='content' else
                            wiki.page(title=page).sections #if type=='sections'
                ,metadata=dict(source=page)#f"Wikipedia://{page}")
            )
        except: return Document(page_content='', metadata=dict(source=page))

    def docs2store(self, docs, split=True):
        docs = RecursiveCharacterTextSplitter(
                chunk_size = 1000, chunk_overlap = 0, length_function = len, # todo specify the Characters ["\n\n", "\n", "."],
            ).split_documents(docs) if split else docs
        return Chroma.from_documents(docs, embeddings)

    def most_relevant(self, query, store, n=3):
        relevants = store.similarity_search_with_score(query)

        return [d[0] for d in sorted(relevants, key=lambda s: s[1], reverse=True)[:n]]


    def find_relevant_articles(self, question):
        searches = self.wiki_queries(question).searches
        results = []
        for s in searches:
            w = wiki.search(s, results=2)
            results.extend(w)

        results = list(set(results))

        summaries = self.docs2store([self.page2doc(r) for r in results], split=False)

        return [d.metadata['source'] for d in self.most_relevant(question, summaries)]

    def answer(self, question):
        pages = self.find_relevant_articles(question)

        store = self.docs2store([self.page2doc(p, type='content') for p in pages])

        relevants = self.most_relevant(question, store)

        # this logic is kinda similar to main, just that we already have computed all alternatives ahead
        for doc in relevants:
            answer = self.answer_atomic(question, doc.page_content)
            if answer.found: return answer.answer, doc

        return False, relevants[0]







# adapted from https://python.langchain.com/en/latest/use_cases/question_answering.html
import readline
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms.openai import OpenAI
from lib.utils import load_doc, repl, get_input_prompt_session
import glob

def main():
    llm = OpenAI(temperature=0, model_name="text-davinci-003")
    print('Provide documents to index (only support txt an pdf files currently)')
    session = get_input_prompt_session('ansired')
    docnames = []
    while True:
        dnames = session.prompt('Document (e.g., "docs/*", enter "exit" when done): ')
        if dnames == 'exit':
            break
        docnames.extend([d for d in glob.glob(dnames) if d.split('.')[-1] in ['txt', 'pdf']])

    loaders = [load_doc(docname) for docname in docnames]
    index = VectorstoreIndexCreator().from_loaders(loaders)
    docnames_str = '\n  '.join([f'{i}: {docname}' for i, docname in enumerate(docnames)])
    print(f"ask me anything about \n  {docnames_str}")

    repl(lambda user_input:
         index.query_with_sources(user_input, llm))


if __name__ == "__main__":
    main()

import re, os, tqdm
from pybtex.database import parse_string
from datetime import datetime
from joblib import Parallel, delayed

import langchain
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.cache import InMemoryCache



def read_bib_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            bib_str = file.read()
            return bib_str
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading the file.\n{e}")
        return None


def read_bibtex(bibtex_str):
    entries = []
    entry_pattern = r'@article{([\w\d_-]+),([^}]*)}'
    fields_pattern = r'\s*([^=]+)\s*=\s*{(.+?)}'

    entries_data = re.findall(entry_pattern, bibtex_str)
    for entry_data in entries_data:
        entry_dict = {}
        entry_dict['key'] = entry_data[0]

        fields_data = re.findall(fields_pattern, entry_data[1])
        for field_data in fields_data:
            field_name = field_data[0].strip().lower()
            field_value = field_data[1].strip()
            entry_dict[field_name] = field_value

        entries.append(entry_dict)

    return entries


def extract_titles_and_filepaths(bibtex_str):
    titles = []
    filepaths = []

    bib_data = parse_string(bibtex_str, 'bibtex')
    for entry_key, entry in bib_data.entries.items():
        if 'title' in entry.fields and 'file' in entry.fields:
            titles.append(entry.fields['title'])
            files = entry.fields['file'].split(';')
            for i in range(len(files)):
                files[i] = files[i].split(':', 1)
            single_list = [item for sublist in files for item in sublist]
            filepaths.append(single_list)

    return titles, filepaths


def is_pdf(filepath):
    _, file_extension = os.path.splitext(filepath)
    return file_extension.lower() == '.pdf'


def process_bibliography(bib, storage):
    bib_str = read_bib_file(bib)
    titles, filepaths = extract_titles_and_filepaths(bib_str)

    new_title = []
    new_file = []

    # only keep first valid filepath for each title
    for i in range(len(titles)):

        # first remove all non pdfs
        filepaths[i] = [f for f in filepaths[i] if is_pdf(f)]
        if len(filepaths[i]) == 0:
            continue

        for j in range(len(filepaths[i])):
            if isinstance(filepaths[i], list):
                filepaths[i][j] = os.path.join(storage, filepaths[i][j])
            else:
                filepaths[i] = os.path.join(storage, filepaths[i])
        valid_filepaths = [filepath for filepath in filepaths[i] if os.path.exists(filepath)]
        if len(valid_filepaths) > 0:
            new_title.append(titles[i])
            new_file.append(valid_filepaths[0])
        else:
            print(titles[i])
            print(filepaths[i])

    return new_title, new_file


def process_bibliography_parallel(bib, storage):
    bib_str = read_bib_file(bib)
    titles, filepaths = extract_titles_and_filepaths(bib_str)

    def process_title(title, filepaths):
        new_filepaths = []
        for filepath in filepaths:
            if isinstance(filepath, list):
                filepath = [os.path.join(storage, fp) for fp in filepath]
            else:
                filepath = os.path.join(storage, filepath)
            valid_filepaths = [fp for fp in filepath if os.path.exists(fp)]
            if valid_filepaths:
                new_filepaths.append(valid_filepaths[0])
        return title, new_filepaths

    results = Parallel(n_jobs=-1)(delayed(process_title)(title, filepaths[i]) for i, title in enumerate(titles))
    new_titles, new_filepaths = zip(*results)

    return list(new_titles), list(new_filepaths)


def get_context(file, question):
    all_context = []
    documents = []

    print("Loading Documents")
    if isinstance(file, list):
        for i in tqdm.tqdm(range(len(file))):
            documents.append(PyPDFLoader(file[i]).load_and_split())
    else:
        documents = [PyPDFLoader(file).load_and_split()]

    print("Computing Embeddings")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    for i in tqdm.tqdm(range(len(documents))):

        texts = text_splitter.split_documents(documents[i])
        faiss_index = FAISS.from_documents(texts, embeddings)
        matched_docs = faiss_index.similarity_search(question, 4)
        context = ""
        for doc in matched_docs:
            context = context + doc.page_content + " \n\n "
        all_context.append(context)

    return all_context


def get_context_parallel(file, question):
    print("Loading Documents")
    if isinstance(file, list):
        documents_list = [PyPDFLoader(f).load_and_split() for f in tqdm.tqdm(file)]
    else:
        documents_list = [PyPDFLoader(file).load_and_split()]

    print("Computing Embeddings")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    def process_document(doc):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts = text_splitter.split_documents(doc)
        faiss_index = FAISS.from_documents(texts, embeddings)
        matched_docs = faiss_index.similarity_search(question, 4)
        context = ""
        for doc in matched_docs:
            context = context + doc.page_content + " \n\n "
        return context

    all_context = Parallel(n_jobs=-1)(delayed(process_document)(doc) for doc in tqdm.tqdm(documents_list))

    return all_context


def get_prompts(template, context):
    prompt = []
    for i in tqdm.tqdm(range(len(context))):
        prompt.append(
            PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context[i]))
    return prompt


def get_prompts_parallel(template, context):
    def process_context(context):
        return PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)

    prompts = Parallel(n_jobs=-1)(delayed(process_context)(ctx) for ctx in tqdm.tqdm(context))
    return prompts


def get_memoryless_output(llm, prompt, question):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    output = llm_chain.run(question)
    return output


def append_response_to_file(response, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(response + '\n\n')


def check_title_in_files(folder, title, question_task):
    txt_files = [f for f in os.listdir(folder) if f.endswith(".txt") and '{0}_'.format(question_task) in f]
    for file_name in txt_files:
        file_path = os.path.join(folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            if title in content:
                return True

    return False


def has_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return any(char.isalnum() for char in content)


def remove_empty_txt_file(path):
    # List all files in the folder
    file_list = os.listdir(path)

    # Iterate through the files and remove those with no text
    for file_name in file_list:
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt') and not has_text(file_path):
            os.remove(file_path)
            print(f"Removed: {file_name}")

def fuse_files(files, output_path):
    with open(output_path, 'w',  encoding='utf-8') as output:
        for file_path in files:
            with open(file_path, 'r',  encoding='utf-8') as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        output.write(line + "\n")
                    elif i > 0 and line:
                        output.write(line + "\n")
                    elif lines[i-1].strip():
                        output.write(line + "\n")

def get_cfg():
    cfg = {
        'storage': r'C:\Users\Tim\OneDrive - James Cook University\Zotero Storage',
        'bib': r'C:\Users\Tim\Documents\Zotero_bibs\growing.bib',
        'model_path': r'C:\Users\Tim\OneDrive - James Cook University\Models\LLM\wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin',
        'question': "What are the challenges mentioned in the following context?",
        'template': """
                         # Please use the following context to answer questions.
                         # Context: {context}
                         #  - -
                         # Question: {question}
                         # Answer: Let's think step by step and answer in detail.""",
        'folder': 'growing',
        'question_task': 'summary',
        'overwrite': False
    }
    return cfg


def ask_pdf(cfg):
    storage = cfg['storage']
    bib = cfg['bib']
    model_path = cfg['model_path']
    question = cfg['question']
    template = cfg['template']
    folder = cfg['folder']
    question_task = cfg['question_task']

    current_time = datetime.now().strftime("%d_%m_%Y_%H%M%S")

    if not os.path.exists('generated_text'):
        os.mkdir('generated_text')
    folder = 'generated_text/{0}'.format(folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    remove_empty_txt_file(folder)

    file_path = os.path.join(folder, f'{question_task}_{current_time}.txt')
    if not os.path.isfile(file_path):
        open(file_path, 'w').close()

    print("Processing {0}, task: {1}".format(os.path.basename(cfg['bib']), cfg['question_task']))
    title, file = process_bibliography(bib, storage)
    print("Getting Context")
    context = get_context(file, question)
    print("Getting prompts")
    prompt = get_prompts(template, context)

    print("Creating model")
    llm = GPT4All(model=model_path, max_tokens=1000, n_predict=1000, verbose=False, repeat_last_n=0, n_batch=8, n_threads=os.cpu_count()/2)
    #llm = GPT4All(model_name=model_path, n_threads=os.cpu_count())
    #llm.model.set_thread_count(os.cpu_count())

    print("Getting output")
    output = []

    for i in tqdm.tqdm(range(len(prompt))):
        print(title[i])

        if not cfg['overwrite'] and check_title_in_files(folder, title[i], question_task):
            output.append([])
            continue

        llm_chain = LLMChain(prompt=prompt[i], llm=llm)
        output.append(llm_chain.run(question))

        print(output[i])
        print()
        append_response_to_file(title[i], file_path)
        append_response_to_file(output[i], file_path)

    current_time = datetime.now().strftime("%d_%m_%Y_%H%M%S")
    fused_file_path = os.path.join(folder, f'{question_task}_fused_{current_time}.txt')
    # get all text files in the current folder which have the question task in them.
    question_task_files = [os.path.join(folder,f) for f in os.listdir(folder) if '{0}_'.format(question_task) in f]
    fuse_files(files=question_task_files, output_path=fused_file_path)


if __name__ == '__main__':

    # summary question
    summary_question = "What are the main ideas and novelty proposed?"
    # challenges question
    challenges_question = "What are the challenges mentioned?"
    # results question
    results_question = "How do the results from the methods discussed compare to the state-of-the-art? How do the performance metrics compare?"
    # research_gaps
    research_question = "What are the research gaps and future research directions?"


    question = [summary_question, challenges_question, results_question, research_question]
    question_task = ['summary', 'challenges', 'results', 'research']

    default_cfg = get_cfg()

    # get summary and challenges for all .bibs in growing
    growing_bib_folder = r'C:\Users\Tim\Documents\Zotero_bibs\growing'
    bib_name = sorted(os.listdir(growing_bib_folder),reverse=True)
    bib = [os.path.join(growing_bib_folder, f) for f in bib_name]

    cfg = list()
    for i in range(len(bib)):
        for j in range(len(question)):
            new_cfg = get_cfg()
            new_cfg['bib'] = bib[i]
            new_cfg['question'] = question[j]
            new_cfg['folder'] = bib_name[i]
            new_cfg['question_task'] = question_task[j]
            cfg.append(new_cfg)

    for i in range(len(cfg)):
        ask_pdf(cfg[i])

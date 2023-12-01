import json
import os
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import embedding
import evaluation
import QAModel
from typing import Dict, List

######################global variables######################
#test number
test_num = 10
#load qa pairs
qa_file_path = 'data/doc2dial/doc2dial_qa_train.csv'
#load json file
doc_file_path = 'data/doc2dial/doc2dial_doc.json'
#chunk size
cs = 500
#chunk overlap
c_overlap = 0
#embedding model
embedding_model = 'sentence-transformers/gtr-t5-base'
#qa model
qa_model = 'google/flan-t5-large'
#chain type
chain_type = 'stuff' #not used now
#how many retrieved docs to use
topk = 1
#failed doc ids
failed_doc_ids = []
#dataset
dataset = 'doc2dial'

#get doc text
# doc_text = doc2dial_doc['doc_data'][domain][doc_id]['doc_text']
############################################################

def load_qa_file(file_path) -> Dict:
    """
    Load doc2dial_qa_train.csv file
    """
    #check if file is csv
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df
    else:
        raise ValueError('File is not csv')


def load_doc_file(file_path) -> Dict:
    """
    Load doc2dial_doc.json file
    """
    #check if file is json
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            doc2dial_doc = json.load(f)
        return doc2dial_doc
    else:
        raise ValueError('File is not json')

def seaerch_doc(doc, db) -> List[Document]:
    query = doc['question']
    docs = db.similarity_search(query)
    return docs

def append_pre_next(idx, idx_list, max_idx):
    if idx == 0:
        idx_list.append(idx)
        idx_list.append(idx+1)
    elif idx == max_idx:
        idx_list.append(idx-1)
        idx_list.append(idx)
    else:
        idx_list.append(idx-1)
        idx_list.append(idx)
        idx_list.append(idx+1)
    return idx_list

def find_pre_next_doc(docs, retrived_doc, num=1):
    max_idx = len(docs)-1
    if num == 1:
        idx_list = []
        idx = docs.index(retrived_doc[0])
        idx_list = append_pre_next(idx, idx_list, max_idx)
        return idx_list
    else:
        top_docs = retrived_doc[:num]
        idx_list = []
        for top_doc in top_docs:
            idx = docs.index(top_doc)
            idx_list = append_pre_next(idx, idx_list, max_idx)
        return idx_list


def RTRBaseline(qa_set, doc2dial_doc, test=True, test_num=test_num, topk=topk, exp_id=None, new_method=False) -> None:
    """
    qa_set: dataframe of qa pairs {'question', 'answer', 'domain', 'doc_id', 'references', 'dial_id'}
    doc2dial_doc: dict (json file)

        get doc text:
        doc_text = doc2dial_doc['doc_data'][domain][doc_id]['doc_text']
    """

    #iterate through qa_set
    print('Runing RTR Baseline...')
    print('Args: test-mode: {}, topk: {}'.format(test, topk))
    if test:
        print('Test number: {}'.format(test_num))
    else:
        print('Experiment id: {}'.format(exp_id))
        print('Exp Args: embedding model: {}, qa model: {}'.format(embedding_model, qa_model))


    if test:
        flan_qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        qa_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    else:
        flan_qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model)
        qa_tokenizer = AutoTokenizer.from_pretrained(qa_model)

    last_doc_name = ''
    result_df = pd.DataFrame(columns=['question', 'answer', 'passage(context)'])
    total_split_docs = 0
    cnt = 0

    for index, row in qa_set.iterrows():
        if test:
            print('Test running: {}'.format(index))
        doc_text = doc2dial_doc['doc_data'][row['domain']][row['doc_id']]['doc_text']

        #get embeddings
        doc_name = row['doc_id']
        # embeddings = HuggingFaceEmbeddings(model_name = embedding_model)
        #check if using same doc
        if doc_name != last_doc_name:
            #build new index
            document = Document(page_content=doc_text, metadata={"source": row['doc_id']})
            split_documents = embedding.split([document], cs=cs, co=c_overlap)
            total_split_docs += len(split_documents)
            cnt += 1

            db = embedding.embedding(split_documents, model=embedding_model)
            last_doc_name = doc_name
            # if not test:
            #     embedding.save_db(db)
            # print('new index created at: ', index)
        # else: 
            #[x] not need load index every time
            #load index
            # db = embedding.load_db(embeddings)
            # print('index loaded at: ', index)

        #seaerch doc
        try:
            result_docs = seaerch_doc(row, db) # list of retrieved documents
        except:
            # if question is Nan, skip
            failed_doc_ids.append(index)
            print('search failed at: ', index)
            continue
        
        if new_method and result_docs is not None:
            #find pre and next doc
            #now fix num to 1, otherwise may larger than max token length
            idx_list = find_pre_next_doc(split_documents, result_docs, num=1)
            result_docs = [split_documents[i] for i in idx_list]
            # if used new method, topk should not 1, but len(result_docs)
            topk = len(result_docs)
        # ref_list = evaluation.get_ref(row, doc2dial_doc) # true references

        #get answer
        model_answer = QAModel.answer_from_local_model(row['question'], 
                                                       result_docs, 
                                                       tokenizer=qa_tokenizer,
                                                       model=flan_qa_model, 
                                                       model_name=qa_model, 
                                                       ct=chain_type, 
                                                       topk=topk)

        #save answer
        example = {}
        example['question'] = row['question']
        example['answer'] = model_answer #[x] should be model result?
        result_docs_list = result_docs[:topk]
        example['passage'] = '\n'.join([doc.page_content for doc in result_docs_list])

        result_df.loc[len(result_df)] = [example['question'], example['answer'], example['passage']]
        
        if test and index == test_num:
            break
        #save checkpoint
        elif not test and index % 5000 == 0 and index != 0:
            checkpoint_dir = 'data/doc2dial/result_{}_cp'.format(exp_id)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint = os.path.join(checkpoint_dir, 'result_{}_cp{}.csv'.format(exp_id, index))
            result_df.to_csv(checkpoint, index=False)
            print('checkpoint saved at: ', index)
        elif not test and index % 1000 == 0 and index != 0:
            print('Running at: ', index)
    
    # write result_df to csv
    if test:
        output_path = 'data/doc2dial/result_test.csv'
    else:
        output_path = 'data/doc2dial/result_{}.csv'.format(exp_id)
    
    result_df.to_csv(output_path, index=False)

    # write result infomation -> txt file
    filename = 'data/doc2dial/result_info.csv'
    if new_method or dataset != 'doc2dial':
        filename = 'data/doc2dial/result_info_1.csv'
    try:
        # Try to open the file in append mode
        new_df = pd.read_csv(filename)
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame
        if new_method or dataset != 'doc2dial':
            new_df = pd.DataFrame(columns=['Experiment id','embed model', 'qa_model', 'test mode', 'topk', 'chunk_size', 'chunk_overlap', 'new_method', 'dataset'])
        else:
            new_df = pd.DataFrame(columns=['embed model', 'qa_model', 'test mode', 'topk', 'chunk_size', 'chunk_overlap'])
    print('='*20)
    print('Total split docs: ', total_split_docs)
    print('Total split times: ', cnt)
    print('Avg split docs: ', total_split_docs/cnt)

    if new_method or dataset != 'doc2dial':
        new_row = [exp_id, embedding_model, qa_model, test, topk, cs, c_overlap, new_method, dataset]
    else:
        new_row = [embedding_model, qa_model, test, topk, cs, c_overlap]
    new_df.loc[len(new_df)] = new_row
    new_df.to_csv(filename, index=False)
    print("Exp info written to the file successfully.")

    # if not test:
    #     fail_filename = 'data/doc2dial/failed_doc_ids_.csv'
    #     with open(fail_filename, 'w') as f:
    #         for item in failed_doc_ids:
    #             f.write("%s\n" % item)

    #evaluation process -> in evaluation.py
    #[x] iterate through answer file
    #[x] compare em, f1?, autoais
    #[x] replace /n with \n 

if __name__ == '__main__':
    print('Running RTR Baseline...')
    start_id = 0
    print('Start id: ', start_id)
    df = load_qa_file(qa_file_path)
    if start_id != 0:
        df = df[start_id:]
    doc2dial_doc = load_doc_file(doc_file_path)
    RTRBaseline(df, doc2dial_doc, test=False, test_num=5, topk=1, exp_id=4, new_method=True)
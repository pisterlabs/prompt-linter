from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import os
import en_core_web_sm
import pickle
import json

def write_to_txt_file(path, data):
    """
    :param

    path: path where to be saved
    data: triples to be written in txt

    """
    f = open(path, "w")
    for i in range(data.shape[0]):
        line = ''
        for j in range(data.shape[1]):
            if(j==0):
                line = str(data[i][j])
            else:
                line = line + '\t' + str(data[i][j])
        f.write(line)
        f.write("\n")
        print(line)
    f.close()

def load_nlp():
    '''
    @param:
    @return: nlp object
    '''
    nlp = en_core_web_sm.load()
    nlp.add_pipe("entityLinker", last=True)
    return nlp

def preprocess_texts(raw_text):
    '''
    @param raw_text: the concatinated text to be processed
    @return texts: the splitted and tokenized text
    '''
    text_splitter = CharacterTextSplitter(
                        separator = "\n",
                        chunk_size = 1024,
                        chunk_overlap  = 200,
                        length_function = len,
                    )
    texts = text_splitter.split_text(raw_text)
    return texts

def read_pdf_text(path, preprocess_langchain=False):
    '''
    @param path: the pdf object path
    @param preprocess_langchain: preprocessing flag from langchain
    @return texts: all the text from the pdf concatinated
    '''
    reader = PdfReader(path)
    raw_text = ''

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    if preprocess_langchain:
        texts = preprocess_texts(raw_text)
    else:
        texts = raw_text
    return texts

def process_all_pdfs(directory_path, preprocess_langchain=False):
    '''
    @param directory_path: get the directory of the documentstore
    @param preprocess_langchain: if the preprocess for langchain to optimize token in chunks should be done
    @param returns: all the concatinated texts from pdfs
    '''
    all_texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory_path, filename)
            texts = read_pdf_text(filepath, preprocess_langchain)
            all_texts.extend(texts)
    return all_texts


def get_linked_entities(nlp, entity):
    # add pipeline (declared through entry_points in setup.py)
    all_linked_entities = nlp(entity)._.linkedEntities
    return all_linked_entities

def store_triples(triple_list, path):
    with open(path, "w") as file:
        for triple in triple_list:
            line = "\t".join(triple)
            file.write(line + "\n")

def obtain_graph_information(triples):
    '''
    @param triples: list of triples
    @return: dictionary of entities and relations
    '''
    entities = set()
    relations = set()
    for triple in triples:
        entities.add(triple[0])
        entities.add(triple[2])
        relations.add(triple[1])
    return entities, relations

def read_json_from_file(filepath):
    '''
    @param filepath: the file path where the json file is located
    @return: the json object
    '''
    with open(filepath, "r") as f:
        return json.load(f)

def read_text_from_file(filepath):
    '''
    @param filepath: the path of the text file to read
    @return: a reader object
    '''
    with open(filepath, 'r') as f:
        return f.read()

def obtain_connection_information(nlp ,entity_list):
    '''
    @param nlp: the nlp object
    @param entity_list:  the list of entities found
    @return: the dictionary containing entities and their connections
    '''
    neighborhood_dict = {}
    for entity in entity_list:
        linked_entities_per_entity = get_linked_entities(nlp, str(entity))
        print(f"entity_label: {entity}")
        print(f"connected_entity_object: {linked_entities_per_entity}")
        print('######################')
        neighborhood_dict[entity] = linked_entities_per_entity
    return neighborhood_dict

# def load_nlp():
#     '''
#     @param:
#     @return: nlp object
#     '''
#     nlp = en_core_web_sm.load()
#     return nlp

# specify your desired json file path
def store_to_json(json_file_path, connection_information):
    '''
    @param json_file_path: path to store the json file
    @param connection_information: the connection information
    @return:
    '''
    # open a file for writing
    with open(json_file_path, 'w') as file:
        # write the Python dictionary to a JSON file
        json.dump(connection_information, file)

def store_to_pickle(pickle_file_path, connection_information):
    '''
    @param pickle_file_path: the path to pkl file
    @param connection_information:  the connection information
    @return: None
    '''
    # open a file for writing
    with open(pickle_file_path, 'wb') as file:
        # write the Python dictionary to a pickle file
        pickle.dump(connection_information, file)

def store_triples(triple_list, path):
    '''
    @param triple_list: the list of triples to store
    @param path: the path to store the triples
    @return: NULL
    '''
    with open(path, "w") as file:
        for triple in triple_list:
            line = "\t".join(triple)
            file.write(line + "\n")
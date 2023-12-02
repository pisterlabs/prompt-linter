from langchain.document_loaders import DirectoryLoader
from unstructured.cleaners.core import clean_extra_whitespace

from typing import Union

import os
import glob

# let's add logging/printing to the load_directory function

ALLOWED_FILES = ['pdf', 'txt', 'docx', 'xml', 'html', 'md', 'json']


def load_html(file_path: str):
    '''
        function to load html files form the dir

        params:
            file_path: str: path of the file with its name

        returns:
            Doc object
    '''

    #import HTML loader
    from langchain.document_loaders import UnstructuredHTMLLoader
    #prepare loader
    print(f"Loading HTML file '{file_path}'...")
    loader = UnstructuredHTMLLoader(file_path)
    #load document
    doc = loader.load()
    print(f"Loaded HTML file '{doc.metadata['filename']}'")
    #return document
    return doc
    

def load_directory(path:str, file_types: Union[str, list]):
    '''
        function to load documents from a directory
    
        params:
            path: str: path pof the dir to load documents from
            file_types: (str or list of strings): file formats to load e.g. .pdf, .docx etc

        returns:
            Docloader object

    '''

    # convert file_types to a list if it is a string
    if isinstance(file_types, str):
        file_types = [file_types]

    # if file_types is not in the allowed files list, raise an error
    for fl in file_types:
        if fl not in ALLOWED_FILES:
            raise ValueError(f"File type {fl} is not allowed. Allowed file types are: {ALLOWED_FILES}")

    #list to store all loader objects
    docs_list = []
    #for each file type in the file_types parameter
    for fl in file_types:
        #load docs
        print(f"Loading {fl} files...")
        try:
            if fl != 'html':
                loader = DirectoryLoader(
                            path,
                            glob=f"**/*.{fl}",
                            show_progress=True,
                            use_multithreading=True,
                            recursive=True
                        )
                docs = loader.load()
                print(f"Loaded {len(docs)} {fl} files")
                #add into the list of loader objects
                docs_list+=docs
            else:
                html_list = glob.glob(os.path.join(path, '*.html'))
                docs = []
                for ht in html_list:
                    #add into the list of loader objects
                    docs_list+=load_html(ht)
                    # docs.append(load_html(ht))

        except Exception as e:
            print(f"Error loading {fl} files: {e}")
    
    return docs_list
    #return the final loaded list

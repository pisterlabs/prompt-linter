from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)

recursive_text_splitter = RecursiveCharacterTextSplitter(        
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    add_start_index = True, # allows timestamp to be obtained
)
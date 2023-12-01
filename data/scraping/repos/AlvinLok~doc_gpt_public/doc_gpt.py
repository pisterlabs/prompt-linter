import streamlit as st
import os
import openai
import shutil
import hashlib

from llama_index import download_loader
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, ServiceContext, set_global_service_context
from llama_index.query_engine import CitationQueryEngine
from llama_index.embeddings import OpenAIEmbedding
from pathlib import Path
from llama_index.llms import OpenAI

st.title("Doc GPT")


try:
    if st.session_state.openai_key:
        st.empty()
    
except:
    st.error('Enter API Key')
    st.session_state.openai_key = None


api_key = st.sidebar.text_input('OpenAI API Key Here:')
if api_key:
    st.session_state.openai_key = api_key

if st.session_state.openai_key:

    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = os.environ["OPENAI_API_KEY"]


    doc_path = './data/'
    temp_folder_path = os.path.join(doc_path, "temp")


    #                     _                   _        _            
    #                    (_)                 | |      | |           
    #   ___  ___  ___ ___ _  ___  _ __    ___| |_ __ _| |_ ___  ___ 
    #  / __|/ _ \/ __/ __| |/ _ \| '_ \  / __| __/ _` | __/ _ \/ __|
    #  \__ \  __/\__ \__ \ | (_) | | | | \__ \ || (_| | ||  __/\__ \
    #  |___/\___||___/___/_|\___/|_| |_| |___/\__\__,_|\__\___||___/  

    if 'response' not in st.session_state:
        st.session_state.response = ''

    if 'file_selected' not in st.session_state:
        st.session_state.file_selected = False

    # flag for checking if the user's action involved selecting a new file, either by upload or by dropdown
    if 'new_file_selected' not in st.session_state:
        st.session_state.new_file_selected = False

    # session state for storing each new uploaded file
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    # session state for storing each new selection in the dropdown
    if 'dropdown_selected_file' not in st.session_state:
        st.session_state.dropdown_selected_file = None

    if 'new_dropdown_file' not in st.session_state:
        st.session_state.new_dropdown_file = False

    if 'new_uploaded_file' not in st.session_state:
        st.session_state.new_uploaded_file = False

    # 0 for upload. 1 for dropdown. Tracks which file the user wants to query from
    if 'focus' not in st.session_state:
        st.session_state.focus = 0

    if 'hash_checker' not in st.session_state:
        st.session_state.hash_checker = False

    # was the generate embeddings button pressed
    if 'embed_button_pressed' not in st.session_state:
        st.session_state.embed_button_pressed = False

    if 'new_file_flag' not in st.session_state:
        st.session_state.new_file_flag = True

    # was the current file selected checked for it's hash
    if 'current_file_hash_checked ' not in st.session_state:
        st.session_state.current_file_hash_checked = False

    # were embeddings created yet for the current selected document
    if 'embeddings_created' not in st.session_state:
        st.session_state.embeddings_created = False

    if 'qa_mode' not in st.session_state:
        st.session_state.qa_mode = True

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

    if "index" not in st.session_state:
        st.session_state.index = None

    if "documents" not in st.session_state:
        st.session_state.documents = None

    if "llm" not in st.session_state:
        st.session_state.llm = "gpt-3.5-turbo-1106"

    if 'reset' not in st.session_state:
        st.session_state.reset = None



    #   ______                _   _                 
    #  |  ____|              | | (_)                
    #  | |__ _   _ _ __   ___| |_ _  ___  _ __  ___ 
    #  |  __| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
    #  | |  | |_| | | | | (__| |_| | (_) | | | \__ \
    #  |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
                                                                                        

    def hard_reset():
        script_directory = './'

        # Clear the contents of the 'data' folder
        data_folder_path = os.path.join(os.path.dirname(script_directory), "data")
        for root, dirs, files in os.walk(data_folder_path):
            for file in files:
                os.remove(os.path.join(root, file))

        # Create a new 'temp' folder within the 'data' folder
        temp_folder_path = os.path.join(data_folder_path, "temp")
        os.makedirs(temp_folder_path, exist_ok=True)

        # Clear the contents of the 'index_storage' folder
        index_storage_path = os.path.join(os.path.dirname(script_directory), "index_storage")
        for root, dirs, files in os.walk(index_storage_path, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir_name in dirs:
                shutil.rmtree(os.path.join(root, dir_name))

        # Delete the existing 'doc_gpt_hashes.txt' file
        doc_gpt_hashes_path = os.path.join(os.path.dirname(script_directory ), "doc_gpt_hashes.txt")
        if os.path.exists(doc_gpt_hashes_path):
            os.remove(doc_gpt_hashes_path)

        # Create a new empty 'doc_gpt_hashes.txt' file
        open(doc_gpt_hashes_path, 'w').close()

    def send_click(index):
        query_engine_settings = {}
        
        if advanced_options:

            for option, value in selected_options.items():
                if value:
                    query_engine_settings.update(query_engine_options[option])
            
            if context_controls:
                query_engine_settings["similarity_top_k"] = context_value

        if citation_query_engine:
            query_engine = CitationQueryEngine.from_args(index, **query_engine_settings, citation_chunk_size=256)
        else:
            query_engine = index.as_query_engine(**query_engine_settings)
        
        response = query_engine.query(st.session_state.prompt)
        st.session_state.response = response


    #calculates hash of current processing document
    def calculate_hash(file_path):
        hash_object = hashlib.sha256()
        with open(file_path, "rb") as file:
            while chunk := file.read(8192):
                hash_object.update(chunk)
        return hash_object.hexdigest()

    # checks if current document already has a hash
    def check_hashes():
        hashes_file_path = "doc_gpt_hashes.txt"

        # Create the hashes file if it doesn't exist
        if not os.path.exists(hashes_file_path):
            open(hashes_file_path, "w").close()


        # Dictionary to store file hashes
        file_hashes = {}

        # Load existing hashes from the file
        with open(hashes_file_path, "r") as hashes_file:
            for line in hashes_file:
                file_name, file_hash = line.strip().split(":")
                file_hashes[file_name] = file_hash

        # Calculate and store hashes for new files
        temp_files = os.listdir(temp_folder_path)
        file_name = temp_files[0]  
        file_path = os.path.join(temp_folder_path, file_name)
        file_hash = calculate_hash(file_path)

        if file_hash not in file_hashes.values():
            file_hashes[file_name + "_" + file_hash] = file_hash
            st.session_state.new_file_flag = True
            st.info("Hash not recognized, new embeddings will be created")
        else:
            st.session_state.new_file_flag = False
            st.info("Hash recognized. Existing embeddings will be used")
        
        # Write all hashes to the hashes file
        with open(hashes_file_path, "w") as hashes_file:
            for file_name, file_hash in file_hashes.items():
                hashes_file.write(f"{file_name}:{file_hash}\n")
        
        st.session_state.current_file_hash_checked == True
        
        return file_hash

    def create_embedding_directory(file_hash):
        return os.path.join("./index_storage", file_hash)

    def get_chat_engine():
        # Check if chat_engine is stored in the session state
        if 'chat_engine' not in st.session_state:

            index = st.session_state.index
            chat_engine = index.as_chat_engine(chat_mode='react', verbose=True)
            st.session_state.chat_engine = chat_engine
        return st.session_state.chat_engine

    def generate_chat_response(prompt):
        chat_engine = get_chat_engine()

        user_input = prompt.strip()
        response = chat_engine.chat(user_input)

        return response.response

    #   _    _                _                             _    _____ _     _      _                
    #  | |  | |              | |                           | |  / ____(_)   | |    | |               
    #  | |__| | ___  __ _  __| | ___ _ __    __ _ _ __   __| | | (___  _  __| | ___| |__   __ _ _ __ 
    #  |  __  |/ _ \/ _` |/ _` |/ _ \ '__|  / _` | '_ \ / _` |  \___ \| |/ _` |/ _ \ '_ \ / _` | '__|
    #  | |  | |  __/ (_| | (_| |  __/ |    | (_| | | | | (_| |  ____) | | (_| |  __/ |_) | (_| | |   
    #  |_|  |_|\___|\__,_|\__,_|\___|_|     \__,_|_| |_|\__,_| |_____/|_|\__,_|\___|_.__/ \__,_|_|   
                                                    




    # Check if the temp folder exists
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)  # Create the folder if it doesn't exist

    temp_files = os.listdir(temp_folder_path)
    sidebar_placeholder = st.sidebar.container()


    #   _      _      __  __    _____ ______ _      ______ _____ _______ _____ ____  _   _ 
    #  | |    | |    |  \/  |  / ____|  ____| |    |  ____/ ____|__   __|_   _/ __ \| \ | |
    #  | |    | |    | \  / | | (___ | |__  | |    | |__ | |       | |    | || |  | |  \| |
    #  | |    | |    | |\/| |  \___ \|  __| | |    |  __|| |       | |    | || |  | | . ` |
    #  | |____| |____| |  | |  ____) | |____| |____| |___| |____   | |   _| || |__| | |\  |
    #  |______|______|_|  |_| |_____/|______|______|______\_____|  |_|  |_____\____/|_| \_|


    new_llm = st.sidebar.text_input("Enter the name of LLM")
    if new_llm:
        st.session_state.llm = new_llm

    st.sidebar.success(f"Selected LLM: {st.session_state.llm}")

    llm_service_context = ServiceContext.from_defaults(
        llm=OpenAI(model=st.session_state.llm)
    )

    #   ______           _              _     _ _               __  __           _      _    _____      _           _   _             
    #  |  ____|         | |            | |   | (_)             |  \/  |         | |    | |  / ____|    | |         | | (_)            
    #  | |__   _ __ ___ | |__   ___  __| | __| |_ _ __   __ _  | \  / | ___   __| | ___| | | (___   ___| | ___  ___| |_ _  ___  _ __  
    #  |  __| | '_ ` _ \| '_ \ / _ \/ _` |/ _` | | '_ \ / _` | | |\/| |/ _ \ / _` |/ _ \ |  \___ \ / _ \ |/ _ \/ __| __| |/ _ \| '_ \ 
    #  | |____| | | | | | |_) |  __/ (_| | (_| | | | | | (_| | | |  | | (_) | (_| |  __/ |  ____) |  __/ |  __/ (__| |_| | (_) | | | |
    #  |______|_| |_| |_|_.__/ \___|\__,_|\__,_|_|_| |_|\__, | |_|  |_|\___/ \__,_|\___|_| |_____/ \___|_|\___|\___|\__|_|\___/|_| |_|
    #                                                    __/ |                                                                        
    #                                                   |___/                                                                         


    query_mode = st.sidebar.selectbox('Select Mode', ['Simple Q&A', 'Chatbot'], index=0)
    if query_mode == 'Simple Q&A':
        st.session_state.qa_mode = True
    else:
        st.session_state.qa_mode = False


    selected_model = st.sidebar.radio("Select Embedding Model", ["bge-large-en", "bge-base-en", "text-embedding-ada-002"])


    if selected_model == "bge-large-en":
        st.sidebar.info("Default model for local vector embedding")
    elif selected_model == "bge-base-en":
        st.sidebar.info("A lighter weight version of bge-large-en")
    elif selected_model == "text-embedding-ada-002":
        st.sidebar.info("OpenAI's embedding based on the ADA-002 model.")

    if selected_model == "text-embedding-ada-002":
        service_context = ServiceContext.from_defaults(embed_model=OpenAIEmbedding())
        st.sidebar.warning("text-embedding-ada-002 is not a local model and requires API calls. This will expose your uploads to OpenAI's servers")
        set_global_service_context(service_context)
    elif selected_model == "bge-large-en":
        service_context = ServiceContext.from_defaults(embed_model="local:BAAI/bge-large-en")
        set_global_service_context(service_context)
    elif selected_model == "bge-base-en":
        service_context = ServiceContext.from_defaults(embed_model="local:BAAI/bge-base-en")
        set_global_service_context(service_context)


    #               _                               _    ____        _   _                 
    #      /\      | |                             | |  / __ \      | | (_)                
    #     /  \   __| |_   ____ _ _ __   ___ ___  __| | | |  | |_ __ | |_ _  ___  _ __  ___ 
    #    / /\ \ / _` \ \ / / _` | '_ \ / __/ _ \/ _` | | |  | | '_ \| __| |/ _ \| '_ \/ __|
    #   / ____ \ (_| |\ V / (_| | | | | (_|  __/ (_| | | |__| | |_) | |_| | (_) | | | \__ \
    #  /_/    \_\__,_| \_/ \__,_|_| |_|\___\___|\__,_|  \____/| .__/ \__|_|\___/|_| |_|___/
    #                                                         | |                          
    #                                                         |_|                          

    citation_query_engine = st.sidebar.toggle("Include Citations", value=True)

    advanced_options = st.sidebar.toggle("Advanced Options", value=False)
    st.sidebar.divider()


    query_engine_options = {
        "deep_summarization": {"response_mode": "tree_summarize"},    
    }

    selected_options = {}

    if advanced_options:

        for option, settings in query_engine_options.items():
            display_name = option.replace("_", " ").title()
            selected_options[option] = st.sidebar.toggle(display_name)

        context_controls = st.sidebar.toggle("Context Controls", value=False)


        context_value = 5
        if context_controls:
            context_value = st.sidebar.slider("Context Value", min_value=1, max_value=10, value=context_value)

    text_area = None
    file_extension = None

    #   ______ _ _        _____                 _____                             _             
    #  |  ____(_) |      |  __ \               |  __ \                           (_)            
    #  | |__   _| | ___  | |__) | __ ___ ______| |__) | __ ___   ___ ___  ___ ___ _ _ __   __ _ 
    #  |  __| | | |/ _ \ |  ___/ '__/ _ \______|  ___/ '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
    #  | |    | | |  __/ | |   | | |  __/      | |   | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
    #  |_|    |_|_|\___| |_|   |_|  \___|      |_|   |_|  \___/ \___\___||___/___/_|_| |_|\__, |
    #                                                                                      __/ |
    #                                                                                     |___/ 
    multiple_files = st.toggle("Analyze multiple files", value=False)
    if multiple_files:
        uploaded_file = st.file_uploader("Choose files", accept_multiple_files = True)
    else:
        uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file and multiple_files == False:
        file_extension = uploaded_file.name.split(".")[-1]

    if st.toggle('Manual Entry'):
        text_area = st.text_area("Enter or paste your text here:")
        st.session_state.file_selected = True


    # check after each pass when a user performs an action whether a new file was uploaded
    if uploaded_file and uploaded_file != st.session_state.uploaded_file:
        st.session_state.new_uploaded_file = True
    else:
        st.session_state.new_uploaded_file = False
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

    # the temp folder is cleared when a new file is uploaded
    if uploaded_file:
        temp_files = os.listdir(temp_folder_path)
        for temp_file in temp_files:
            os.remove(os.path.join(temp_folder_path, temp_file))
        selected_existing_file = '' 

    existing_doc_files = os.listdir(doc_path)
    existing_doc_files = [file for file in existing_doc_files if os.path.isfile(os.path.join(doc_path, file))]

    existing_doc_files = [''] + existing_doc_files

    selected_existing_file = st.selectbox("Select an existing document", existing_doc_files)

    if selected_existing_file:
        file_extension = selected_existing_file.split(".")[-1]

    if selected_existing_file and selected_existing_file != st.session_state.dropdown_selected_file:
        st.session_state.new_dropdown_file = True
    else:
        st.session_state.new_dropdown_file = False
    if selected_existing_file:
        st.session_state.dropdown_selected_file = selected_existing_file

    st.session_state.new_file_selected = st.session_state.new_dropdown_file or st.session_state.new_uploaded_file

    if st.session_state.new_dropdown_file or st.session_state.new_uploaded_file:
        if st.session_state.new_dropdown_file:
            st.session_state.focus = 1
        else:
            st.session_state.focus = 0
        
    if uploaded_file is None or uploaded_file == '':
        st.session_state.focus = 1

    # add variable to check the last time the the file changed, which one was it

    if uploaded_file:
        selected_existing_file = '' 

    temp_folder_path = os.path.join(doc_path, "temp")
    temp_files = os.listdir(temp_folder_path)
    for temp_file in temp_files:
        os.remove(os.path.join(temp_folder_path, temp_file))

    prev_selected_existing_file = ""

    # Clear existing documents in the temporary folder only when a new selection is made
    if prev_selected_existing_file != selected_existing_file:
        temp_files = os.listdir(temp_folder_path)
        for temp_file in temp_files:
            os.remove(os.path.join(temp_folder_path, temp_file))

        prev_selected_existing_file = selected_existing_file

    if uploaded_file is not None or selected_existing_file != "":
        st.session_state.file_selected = True

    if st.session_state.new_uploaded_file is None and selected_existing_file != "":
        st.session_state.new_file_selected = True

    # forces user to generate embeddings
    if st.session_state.new_file_selected:
        st.session_state.current_file_hash_checked = False
        st.session_state.embeddings_created = False
        st.session_state.embed_button_pressed = False

        # if the user chooses a new file, then the index is wiped and needs to be rewritten
        st.session_state.index = None


    #  __      __       _             _          _   _                               _   _    _           _        _____ _               _    
    #  \ \    / /      | |           (_)        | | (_)                             | | | |  | |         | |      / ____| |             | |   
    #   \ \  / /__  ___| |_ ___  _ __ _ ______ _| |_ _  ___  _ __     __ _ _ __   __| | | |__| | __ _ ___| |__   | |    | |__   ___  ___| | __
    #    \ \/ / _ \/ __| __/ _ \| '__| |_  / _` | __| |/ _ \| '_ \   / _` | '_ \ / _` | |  __  |/ _` / __| '_ \  | |    | '_ \ / _ \/ __| |/ /
    #     \  /  __/ (__| || (_) | |  | |/ / (_| | |_| | (_) | | | | | (_| | | | | (_| | | |  | | (_| \__ \ | | | | |____| | | |  __/ (__|   < 
    #      \/ \___|\___|\__\___/|_|  |_/___\__,_|\__|_|\___/|_| |_|  \__,_|_| |_|\__,_| |_|  |_|\__,_|___/_| |_|  \_____|_| |_|\___|\___|_|\_\


    if st.session_state.file_selected or st.session_state.embed_button_pressed:
        embedding_button = st.button('Generate Embeddings')
        
        if embedding_button or st.session_state.embed_button_pressed:
            st.session_state.embed_button_pressed = True
            if st.session_state.index is None:
                with st.spinner("Generating Embeddings..."):

                    
                    #  _                 _   ___ _ _        
                    # | |   ___  __ _ __| | | __(_) |___ ___
                    # | |__/ _ \/ _` / _` | | _|| | / -_|_-<
                    # |____\___/\__,_\__,_| |_| |_|_\___/__/

                    # clear temp folder
                    temp_files = os.listdir(temp_folder_path)
                    for temp_file in temp_files:
                        os.remove(os.path.join(temp_folder_path, temp_file))
                    
                    # load uploaded file into temp folder and data folder if the focus is 0, which means upload
                    if st.session_state.focus == 0:
                        temp_files = os.listdir(temp_folder_path)
                        for temp_file in temp_files:
                            os.remove(os.path.join(temp_folder_path, temp_file))

                        if multiple_files:
                            # if multiple files are being used, then load all into temp file
                            for file in uploaded_file:  # Assume 'uploaded_files' is a list of uploaded files
                                bytes_data = file.read()
                                with open(os.path.join(temp_folder_path, file.name), 'wb') as f:
                                    f.write(bytes_data)

                                with open(os.path.join(doc_path, file.name), 'wb') as f:
                                    f.write(bytes_data)
                        else:

                            bytes_data = uploaded_file.read()
                            with open(os.path.join(temp_folder_path, uploaded_file.name), 'wb') as f:
                                f.write(bytes_data)
                            
                            with open(os.path.join(doc_path, uploaded_file.name), 'wb') as f:
                                f.write(bytes_data)
                            file_name = uploaded_file.name

                    # load selected file into temp folder and data folder if the focus is 1
                    elif st.session_state.focus == 1:
                        if text_area is None:
                            temp_files = os.listdir(temp_folder_path)
                            for temp_file in temp_files:
                                os.remove(os.path.join(temp_folder_path, temp_file))
                            src_path = os.path.join(doc_path, selected_existing_file)
                            dest_path = os.path.join(temp_folder_path, selected_existing_file)
                            shutil.copyfile(src_path, dest_path)

                        file_name = selected_existing_file


                    # checks if there is more than one file in the temp folder
                    if multiple_files == False:
                        if len(temp_files) > 1:
                            st.error("Error: More than one file found in the temporary folder.")
                            st.stop()  # Stop the execution of the app

                    # _           _                         _                     __ _ _     
                    # | |_ _____ _| |_   __ _ _ _ ___ __ _  | |_ ___ _ __  _ __   / _(_) |___ 
                    # |  _/ -_) \ /  _| / _` | '_/ -_) _` | |  _/ -_) '  \| '_ \ |  _| | / -_)
                    # \__\___/_\_\\__| \__,_|_| \___\__,_|  \__\___|_|_|_| .__/ |_| |_|_\___|
                    #                                                     |_|                 


                    if text_area is not None:

                        text_content = text_area

                        # Generate a unique file name (you can modify this as needed)
                        file_name = "user_text.txt"

                        # Create the full path to the text file
                        file_path = os.path.join(temp_folder_path, file_name)

                        # Write the text content to the text file
                        with open(file_path, "w") as text_file:
                            text_file.write(text_content)


                    #  ___ _           _     _  _         _           
                    # / __| |_  ___ __| |__ | || |__ _ __| |_  ___ ___
                    #| (__| ' \/ -_) _| / / | __ / _` (_-< ' \/ -_|_-<
                    # \___|_||_\___\__|_\_\ |_||_\__,_/__/_||_\___/__/

                    # # check hashes only if the current action is selecting a new file
                    if multiple_files:
                        st.session_state.current_file_hash_checked = True

                        documents = SimpleDirectoryReader(temp_folder_path).load_data()
                    else:
                        if st.session_state.current_file_hash_checked == False:
                            if text_area is None:
                                check_hashes()

                            file_path = os.path.join(temp_folder_path, file_name)

                            file_hash = calculate_hash(file_path)

                        embedding_dir = create_embedding_directory(file_hash)


                        if file_extension == 'pptx':
                            temp_files = os.listdir(temp_folder_path)
                            file_name = temp_files[0]

                            # Construct the full path to the file
                            file_path = os.path.join(temp_folder_path, file_name)
                            PptxReader = download_loader("PptxReader")
                            loader = PptxReader()
                            documents = loader.load_data(file=Path(file_path))
                        else:
                            documents = SimpleDirectoryReader(temp_folder_path).load_data()

                    st.session_state.documents = documents

                    
                    if st.session_state.new_file_flag:
                        if multiple_files == False:
                            os.makedirs(embedding_dir, exist_ok=True)

                        
                        index = VectorStoreIndex.from_documents(documents, service_context=llm_service_context)
                        if multiple_files == False:
                            index.storage_context.persist(persist_dir=embedding_dir)
                        st.session_state.index = index

                    else:
                        storage_context = StorageContext.from_defaults(persist_dir=embedding_dir)
                        index = load_index_from_storage(storage_context, service_context=llm_service_context)
                        st.session_state.index = index

                st.session_state.embeddings_created = True

                st.session_state.index = index

            if multiple_files == False:
                sidebar_placeholder.header('Current Processing Document:')
                selected_doc_name = uploaded_file.name if st.session_state.focus == 0 else selected_existing_file
                sidebar_placeholder.subheader(selected_doc_name)
                sidebar_placeholder.write(st.session_state.documents[0].get_text()[:500]+'....')

    #    ____                          ______                _   _                _____      _ _ 
    #   / __ \                        |  ____|              | | (_)              / ____|    | | |
    #  | |  | |_   _  ___ _ __ _   _  | |__ _   _ _ __   ___| |_ _  ___  _ __   | |     __ _| | |
    #  | |  | | | | |/ _ \ '__| | | | |  __| | | | '_ \ / __| __| |/ _ \| '_ \  | |    / _` | | |
    #  | |__| | |_| |  __/ |  | |_| | | |  | |_| | | | | (__| |_| | (_) | | | | | |___| (_| | | |
    #   \___\_\\__,_|\___|_|   \__, | |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|  \_____\__,_|_|_|
    #                           __/ |                                                            
    #                          |___/                                                             

    if st.session_state.index != None and st.session_state.embeddings_created and st.session_state.qa_mode:


        with st.form(key='my_form'):
            prompt_input = st.text_input("Ask something: ", key='prompt')
            send_button = st.form_submit_button("Send")

        if send_button:
            send_click(st.session_state.index)

        if st.session_state.response:
            st.subheader("Response: ")
            st.success(st.session_state.response, icon="🤖")
        
            if citation_query_engine:
                response = st.session_state.response
                for i, node in enumerate(response.source_nodes):
                    expander_title = f"Source {i + 1}" 
                    with st.expander(expander_title):
                        st.write(node.node.get_text())
            if st.button("Save as Text Document"):
                # Create a text document containing the response and sources
                document_text = str(st.session_state.response) + "\n\n"
                if citation_query_engine:
                    for i, node in enumerate(response.source_nodes):
                        document_text += f"Source {i + 1}:\n{node.node.get_text()}\n\n"

                # Offer the file download to the user
                st.download_button(
                    label="Click to download text document",
                    data=document_text,
                    key="text_document",
                    file_name="saved_document.txt",
                    mime="text/plain",
                )
                st.success("Text document saved successfully.")

    if st.session_state.index != None and st.session_state.embeddings_created and st.session_state.qa_mode == False:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_chat_response(prompt) 
                    st.write(response) 
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
        

    if st.button("Reset - Erase all memory and history"):
        st.session_state.reset = True
    if st.session_state.reset == True:
        if st.button("Are you sure?"):
        
            hard_reset()
            st.session_state.reset = None
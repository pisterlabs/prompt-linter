import tempfile
import streamlit as st
import os
from streamlit_ace import st_ace
import session_state_management
import toml
from streamlit_extras.stateful_button import button
from streamlit_extras.add_vertical_space import add_vertical_space
from simple_auth import logout
import openai
import time
import os
from glob import glob
import pandas as pd


def update_text_file(file, value):
    """
    On Change, this saves the current labels to a file called labels.md in the root folder.
    """
    
    with open(file, 'w') as f:
        f.write(value)
    return None

# MIGRATED
def upload_custom_llm_file():
    """
    This function allows the user to upload a custom LLM file.
    """
    # Ask the user to upload a file
    uploaded_file = st.file_uploader("Upload a file", type=['py'])
    # If a file was uploaded, create a df2 dataframe
    if uploaded_file is not None:
        # Get the file extension
        file_extension = uploaded_file.name.split('.')[-1]

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_file_path = os.path.join(tmp_folder, uploaded_file.name)
            with open(tmp_file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Verify the functions in the file
            function_dict = {'function_name': 'custom_llm_output', 'args': ['input', 'max_tokens', 'temperature', 'model', 'functions']}
            if verify_functions(tmp_file_path, function_dict):
                success_message = st.empty()
                success_message.success('Functions verified successfully, you can now save the file by clicking the button below')
                if button('Save Custom LLM', key='save_custom_llm'):
                    success_message.empty()
                    # Get the typebuild root directory from the session state
                    typebuild_root = st.session_state.typebuild_root
                    file_path = os.path.join(typebuild_root, 'custom_llm.py')

                    # if the file already exists, ask the user if they want to overwrite it or not
                    if os.path.exists(file_path):
                        add_vertical_space(2)
                        overwrite = st.radio('File already exists, do you want to overwrite it?', ['Yes', 'No'], index=1)
                        if overwrite == 'Yes':
                            with st.spinner('Saving file...'):
                                time.sleep(2)
                                # Save the file to the data folder
                                with open(file_path, 'wb') as f:
                                    f.write(uploaded_file.getbuffer())
                                st.success(f'File saved successfully')
                        else:
                            st.warning('File not saved')
                    else:
                        # Save the file to the data folder
                        with open(file_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        st.success(f'File saved successfully')
    st.stop()
    return None

import ast
# MIGRATED
def verify_functions(file_path, function_dict):
    # Parse the Python file using the ast library
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())


    # Extract all function definitions from the parsed tree
    functions = []
    for node in ast.walk(tree):
        tmp_dict = {}
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            args = [arg.arg for arg in node.args.args]
            tmp_dict['function_name'] = function_name
            tmp_dict['args'] = args
            functions.append(tmp_dict)

    # Check if the get_llm_output function is present in the functions list
    # if get_llm_output function present and the args do not match, show an error
        with open(os.path.join(st.session_state.dir_path, 'plugins', 'llms.py'), 'r') as f:
            tree = f.read()

    if 'custom_llm_output' not in [i['function_name'] for i in functions]:
        st.error('custom_llm_output function not found in the file, you need to have a custom_llm_output function in the file, for the reference, see the custom_llm_output function in the code block below')
        st.code(tree, language='python')
        st.stop()
    else:
        # Get the args for the get_llm_output function
        custom_llm_output_args = [i['args'] for i in functions if i['function_name'] == 'custom_llm_output'][0]
        # If the args do not match, show an error.  use a set to compare the args
        if set(custom_llm_output_args) != set(function_dict['args']):
            st.error(f'custom_llm_output function args do not match. Expected: {function_dict["args"]}, Actual: {custom_llm_output_args}')
            st.code(tree, language='python')
            st.stop()

        else:
            return True

# MIGRATED
def config_project():
    """
    For a new project, there should be a config.json file in the project_settings folder. if not, then this function will create one.
    this config file should have the following keys:
    - preferred model (str): The preferred model for the project, e.g. gpt-3.5-turbo-16k, gpt-3.5-turbo, gpt-4, etc.
    - api key (str): The API key for the openai API, if preferred model is openai's
    - function_call_availabilty (bool): Does the user have access to the 0613 models of openai?

    Save the api key to streamlit secrets.toml file
    
    """
    # Get the secrets_file_path from the session state
    secrets_file_path = st.session_state.secrets_file_path
    with open(secrets_file_path, 'r') as f:
        config = toml.load(f)
        st.session_state.config = config

    # Check if the user wants to upload a custom LLM file or set the openai API key
    default_index = 0
    api_key = set_or_get_llm_keys()
    if os.path.exists(os.path.join(st.session_state.typebuild_root, 'custom_llm.py')):
        default_index = 1

    llm_selection = st.radio('Select an option', ['Set OpenAI API key', 'Upload Custom LLM'], horizontal=True, index=default_index)

    if llm_selection == 'Upload Custom LLM':
        upload_custom_llm_file()
        st.stop()
    else:
        api_key = st.text_input('Enter OpenAI key', value=st.session_state.config.get('openai', {}).get('key', ''))
        claude_key = st.text_input('Enter Claude key', value=st.session_state.config.get('claude', {}).get('key', ''))

        function_call_availabilty = st.checkbox(
            "(Expert setting) I have access to function calling", 
            value=st.session_state.config.get('function_call_availabilty', True),
            help="Do you have access to openai models ending in 0613? they have a feature called function calling.",
            )
            
        if st.button("Submit config"):
            if api_key == '':
                st.error('Enter the API key')
                st.stop()
            # Save the config to the config.json file
            config = {}
            # set the openai key
            openai.api_key = api_key
            # Save the API key in the secrets module
            config['openai'] = {'key': api_key}
            if claude_key != '':
                config['claude'] = {'key': claude_key}
            config['function_call_availabilty'] = function_call_availabilty
            if function_call_availabilty:
                st.session_state.function_call_type = 'auto'
            else:
                st.session_state.function_call_type = 'manual'
            # Save the config to the config.json file
            with st.spinner('Saving config...'):
                time.sleep(2)

            if not os.path.exists(secrets_file_path):
                with open(secrets_file_path, 'w') as f:
                    f.write('')
            with open(secrets_file_path, 'r') as f:
                config_ = toml.load(f)

            # Add the API key to the config dictionary
            config_['openai'] = {'key': api_key}
            if claude_key != '':
                config_['claude'] = {'key': claude_key}
            config_['function_call_availabilty'] = function_call_availabilty
            # Save the config to the secrets.toml file
            with open(secrets_file_path, 'w') as f:
                toml.dump(config_, f)
                st.toast('Hip!')
                time.sleep(.5)
                st.success('Config saved successfully')

def text_areas(file, key, widget_label):
    """
    We have stored text that the user can edit.  
    Given a file, load the text from the file and display it in a text area.
    On change, save the text to the file.
    """
    # Get the directory and create it if it does not exist
    directory = os.path.dirname(file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Load current value from file.  If not create empty file
    if not os.path.exists(file):
        with open(file, 'w') as f:
            f.write('')
    with open(file, 'r') as f:
        value = f.read()
    
    out = st_ace(
        value=value,
        placeholder="Write your requirements here...",
        language="markdown",
        theme="github",
        show_gutter=False,
        font_size=14,
        wrap=True,
        keybinding="vscode",
        key=f"{key}_{st.session_state.project_folder}",
        )
    
    if out != value:
        update_text_file(file, value=out)
    # Add user requirements to session state
    st.session_state.user_requirements = out
    return out

def extract_python_code(text):
    """
    Extracts Python code snippets from within triple backticks in the given text.
    
    Parameters:
    -----------
    text: str
        The text to search for Python code snippets.
    
    Returns:
    --------
    A list of Python code snippets (strings).
    """
    snippets = []
    in_code_block = False
    for line in text.split('\n'):
        if line.strip() == '```python':
            in_code_block = True
            code = ''
        elif line.strip() == '```' and in_code_block:
            in_code_block = False
            snippets.append(code)
        elif in_code_block:
            code += line + '\n'
    
    # Evaluate code snippets and return python objects
    correct_code = []
    for i, snippet in enumerate(snippets):
        try:
            correct_code.append(eval(snippet))
        except:
            pass
    return correct_code

def get_approved_libraries():
    return "import streamlit as st\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom glob import glob\nimport datetime\nimport faker\nfrom faker import Faker, import faker-commerce"

#--------------CODE TO RUN AT THE START OF THE APP----------------
def set_function_calling_availability(toggle=False):
    """
    This sets the function calling availability to the session state.
    If function_call is not in session state, it looks at secrets.

    Args:
    - toggle: bool, whether to toggle the function calling availability.
    """

    # Get the secrets file path
    secrets_file_path = st.session_state.secrets_file_path

    # Look at the secrets if function_call is not in session state
    if 'function_call' not in st.session_state:
        if not os.path.exists(secrets_file_path):
            st.session_state.function_call = False
        else:
            with open(secrets_file_path, 'r') as f:
                config = toml.load(f)
            if config != {}:
                if 'function_call_availabilty' in config:
                    st.session_state.function_call = config['function_call_availabilty']
                else:
                    st.session_state.function_call = False
            else:
                st.session_state.function_call = False

    # Toggle the function calling availability
    if toggle:
        st.session_state.function_call = not st.session_state.function_call

    return None

def create_user_folder():
    """
    Creates a user folder in the .typebuild folder.
    """
    # Get the user folder from the session state
    if 'user_folder' not in st.session_state:
        home_dir = os.path.expanduser("~")
        st.session_state.user_folder = os.path.join(home_dir, ".typebuild", 'users' ,st.session_state.token)
    
    return None

# MIGRATED
def set_or_get_llm_keys():

    # Check if the user has a secrets file and openai key in the secrets.toml file. if yes, then set the openai key

    # Get the project folder from the session state
    user_folder = st.session_state.user_folder
    # Create the secrets.toml file if it does not exist
    secrets_file_path = os.path.join(user_folder, 'secrets.toml')
    if not os.path.exists(secrets_file_path):
        with open(secrets_file_path, 'w') as f:
            f.write('')
        st.session_state.config = {}
    else:
        with open(secrets_file_path, 'r') as f:
            config = toml.load(f)
            st.session_state.config = config
    api_key = st.session_state.config.get('openai', {}).get('key', '')
    claude_key = st.session_state.config.get('claude', {}).get('key', '')
    if api_key != '':
        openai.api_key = api_key
    if claude_key != '':
        st.session_state.claude_key = claude_key
    return api_key

def starter_code():
    """
    Functions that need to be run at the top of the app.
    """
    # Add all default session states
    session_state_management.main()
    set_or_get_llm_keys()
    
    # Menu bar and other settings
    from menu import get_menu
    new_menu = get_menu()
    
    if st.session_state.new_menu == 'logout':
        logout()
    # create_user_folder()
    set_function_calling_availability()
    if 'upgrade' not in st.session_state:
        temp_upgrade()
        st.session_state.upgrade = True
    return None
    
def extract_list_of_dicts_from_string(res):
    """
    Extracts a list of dictionaries from a string.
    """
    # Remove the new line characters
    res = res.replace('\n', ' ')

    # assuming `res` is the string with the list of dicts
    start_index = res.index('[') 
    end_index = res.rindex(']') + 1
    list_of_dicts_str = res[start_index:end_index]

    
    return eval(list_of_dicts_str)

def temp_upgrade():
    """
    Use this for upgrades
    """
    user_folder = st.session_state.user_folder
    if user_folder.endswith(os.path.sep):
        user_folder = user_folder[:-1]

    projects = glob(os.path.join(user_folder, '**', ''))
     
    for project in projects:
        project_research_file = os.path.join(project, 'research_projects_with_llm.parquet')
        if not os.path.exists(project_research_file):
            create_research_data_file(project)
        else:
            df = pd.read_parquet(project_research_file)
            if 'research_name' not in df.columns:
                create_research_data_file(project)
        

    return None

def x(arr):
    file, col = arr
    file = os.path.basename(file).replace('.parquet', '').replace('_', ' ').title()
    col = col.replace('llm_', '').replace('_', ' ').upper()
    return f"{col} {file}"

def sys_ins_get(row):
    col, file = row
    file = os.path.basename(file).replace('.parquet', '')
    project = os.path.join(*file.split(os.path.sep)[1:-1])
    sys_ins = f"{project}{file}_{col}_sys_ins.txt"
    
    if os.path.exists(sys_ins):
        with open(sys_ins, 'r') as f:
            text = f.read()
    else:
        text = ''
    
    return text

def sys_ins_get(row):
    col, file = row
    file_name = os.path.basename(file)
    file_name_without_extension = os.path.splitext(file_name)[0]
    project_path = os.path.dirname(file_name_without_extension)
    sys_ins = os.path.normpath(os.path.join(project_path, f"{file_name_without_extension}_{col}_sys_ins.txt"))

    if os.path.exists(sys_ins):
        with open(sys_ins, 'r') as f:
            text = f.read()
    else:
        text = ''
    return text


def create_research_data_file(project):
    """
    Create a data model if it does not exist.  Temp upgrade from 0.0.22 to 0.0.23
    """

    data_model = os.path.join(project, 'data_model.parquet')
    
    st.sidebar.warning(f"No data model for {project}")
    if not os.path.exists(data_model):
        return None
    df = pd.read_parquet(data_model)
    
    res_projects = df[df.column_name.str.contains('llm')][['column_name', 'file_name']]
        
    if len(res_projects) == 0:
        res_projects = pd.DataFrame(columns=['research_name', 'project_name', 'file_name', 'input_col', 'output_col', 'word_limit', 'row_by_row', 'system_instruction'])
    else:
        res_projects = res_projects.rename(columns={'column_name': 'output_col'})
        
        res_projects['input_col'] = 'SELECT'
        
        res_projects['research_name'] = res_projects[['file_name', 'output_col']].apply(x, axis=1)

        res_projects['project_name'] = project
        st.sidebar.warning(project)
        res_projects = res_projects.reset_index(drop=True)
        
        res_projects['word_limit'] = 1000

        res_projects['system_instruction'] = ''
        
        res_projects.loc[res_projects.input_col.str.contains('conso'), 'row_by_row'] = True
        
        res_projects.row_by_row = res_projects.row_by_row.fillna(False)
        
        res_projects = res_projects[['project_name', 'research_name', 'file_name', 'input_col', 'output_col', 'word_limit', 'row_by_row', 'system_instruction']]
    
        res_projects['system_instruction'] = res_projects[['output_col', 'file_name']].apply(sys_ins_get, axis=1)
    
    project_research_file = f"{project}research_projects_with_llm.parquet"    
    res_projects.to_parquet(project_research_file, index=False)
    
    return None
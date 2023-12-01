#!/usr/bin/env python3
#
##########################################################################################################
#
# Copyright 2023 LUCIDIO KUHN
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
##########################################################################################################
#
# Requirements:
# * Install Java: If you have not done so yet, install it from https://java.com/en/download/
# * Install PlantUML from https://plantuml.com/download
# * Create and environment variable PLANT_UML_PATH with the full path to the PlantUML.jar file
# * Get API Key from openai: https://platform.openai.com/account/api-keys
# * Create an environment variable OPENAI_API_KEY with the API Key.
# * Install Python: If you have not done so yet, install it from https://www.python.org/downloads/
# * pip install openai
# * pip install streamlit
#
##########################################################################################################
#
# Description:
#   This script uses ChatGPT, PlantUML and StreamLit to bring basic answers to system design questions.
#
##########################################################################################################
#
# How to run:
#   Open a command prompt, go to this folder then run:
#   streamlit run .\systemDesignInterview.py
#
##########################################################################################################
#
# How to debug:
#   Add the following to your launch.json:
#        {
#            "name": "Streamlit",
#            "type": "python",
#            "request": "launch",
#            "program": "C:/Users/lucid/anaconda3/Scripts/streamlit.exe",
#            "args": [
#                 "run",
#                 "${file}",
#                 "--server.port",
#                 "8501"
#            ]
#        }   
#   For you to edit your launch.json, do this:
#     Select the Debug tab
#     Click on the little cog icon at the top.
#   Notice that the "program" path must use forward slashes, even if it is in Windows.
#   The "program" path will be different depending on your environment. 
#   The best way to find out where StreamLit is installed is by opening a command prompt and running:
#     In Linux/Mac:
#       which streamlit
#     In Windows:
#       where.exe streamlit.exe
#   Now, to debug your StreamLit program, select the Debug Tab, select "Streamlit"  the click the green debug button.
#
##########################################################################################################
import openai
import streamlit as st
import tempfile
import os
import re
import subprocess
from time import sleep,time
from typing import List
from colorama import Fore, Style
from collections.abc import Iterable

def printColor(color:object, s:str) -> None:
    print(color, s)
    print(Style.RESET_ALL)    

def printR(s:str)                     -> None: printColor(Fore.RED, s)
def printG(s:str)                     -> None: printColor(Fore.GREEN, s)
def printB(s:str)                     -> None: printColor(Fore.BLUE, s)
def printY(s:str)                     -> None: printColor(Fore.YELLOW, s)
def printM(s:str)                     -> None: printColor(Fore.MAGENTA, s)
def cleanString(s:str, character:str) -> str: return re.sub('[^A-Za-z]+', character, s.strip())
def getTempFilePath(title:str)        -> str: return os.path.join(tempfile.gettempdir(), f'{title}.txt')
def changeExtension(file_path:str, new_extension: str) -> str: return f"{os.path.splitext(file_path)[0]}.{new_extension.strip('.')}"

def openTempFile(title:str, mode:str) -> object:
    file_path = getTempFilePath(title)
    if mode == 'r' and not os.path.exists(file_path):
        return None
    return open(file_path, mode)

def saveInTempFolder(title:str,text:str) -> str: 
    with openTempFile(title, 'w') as f:
        f.write(text)
    return f.name

def readFromTempFolderIfExists(title:str) -> str:
    try:
        with openTempFile(title, 'r') as f:
            return f.read()
    except Exception:
        return None

def queryAIUntilDone(what:str) -> str:
    """
    Summary:
        Sends the current Prompt to ChatGPT and retrieves answer, with possible retries.
    Args:
        what (str): The text being injected into the Prompt. It is used only to check if the ChatGPT answer is already locally cached.

    Returns:
        str: The answer from ChatGPT.

    Important:
        This function also adds the answer from ChatGPT to the current Prompt.
    """
    cache_title   = cleanString(f"{st.session_state['CACHE_ID']}_{what}", "_").lower()
    answer        = ''
    if cached:= readFromTempFolderIfExists(cache_title):
        printB('From cache:')
        printY(cached)
        answer    = cached
    else:
        printB('From OpenAI:')
        with st.spinner(f'Loading from OpenAI: {what}...'):
            for i in range(4):
                try:
                    response = openai.Completion.create(model="text-davinci-003",prompt=st.session_state['PROMPT'],temperature=1,max_tokens=1024,top_p=1,frequency_penalty=0,presence_penalty=0)
                    printY(response.choices[0].text)
                    break
                except Exception as ex:
                    printM(f'Exception trying to read from openai {ex} waiting 5 seconds, {4-i} retries...')
                    sleep(5)
        answer  = response.choices[0].text
    st.session_state['PROMPT'] += answer
    st.session_state['PROMPT'] = ''.join([c for c in st.session_state['PROMPT'] if ord(c) > 0 and ord(c) < 256])
    saveInTempFolder(cache_title, answer)
    return answer

def addReloadButton() -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.text('If diagrams had issues, edit the text then click Reload')
    with col2:
        st.button('Reload', on_click=reloadDiagrams, key=f"reload{int(time())}")

def createThenPlaceUMLImage(text_file_path:str, diagram_type:str) -> None:
    """
    Summary:
        Run PromptUML to convert the UML diagram text to a PNG with the diagram, then place the PNG on the page.
    Args:
        text_file_path (str): File with the UML diagram text in the PromptUML format.
        diagram_type (str): Type of diagram, such as Class Diagram or Sequence Diagram.
    """
    picture_file_path = f'{diagram_type}.png'
    printY(f"java -jar {os.environ['PLANT_UML_PATH']} {text_file_path} {picture_file_path}")
    subprocess.call(['java', '-jar', os.environ['PLANT_UML_PATH'], text_file_path, picture_file_path])
    st.divider()
    st.subheader(diagram_type)    
    st.image(picture_file_path)

def reloadDiagrams() -> None:
    """
    Summary:
        Place the Text items with the locally cached text, then rebuild the UML diagrams with the cached PlantUML texts that were potentially reviewed by the user.
    """
    placeDesignItems('.*Text')
    createThenPlaceUMLImage('Sequence_Diagram.txt', 'Sequence_Diagram')
    createThenPlaceUMLImage('Class_Diagram.txt', 'Class_Diagram')
    addReloadButton()

def placeUMLDiagram(diagram_UML_text:str, diagram_type:str) -> None:
    """
    Summary:
        Given a diagram UML text following the PlantUML notation and the diagram type, create a file with the diagram UML text and kick the function that converts the text to UML image then place the image on the UI.
    Args:
        diagram_UML_text (str): Diagram UML text following Plant UML notation.
        diagram_type (str): The diagram type, such as 'Sequence Diagram' or 'Class Diagram'.
    """
    diagram_type      = cleanString(diagram_type, '_')
    text_file_path    = f'{diagram_type}.txt'
    with open(text_file_path,'w') as f: f.write(diagram_UML_text)
    createThenPlaceUMLImage(text_file_path, diagram_type)

def flattenList(l:Iterable[object]) -> List[object]:
    """
    Summary:
        Given a list with potential sublists, return a "flattened" list.
    Args:
        l (List[object]): List to be flattened.

    Returns:
        List[object]: The flattened list.
    Example:
        flattenList([1, [2, [3,4]]])
            [1, 2, 3, 4]
        flattenList({1, (2, (3,4))}) 
            [1, 2, 3, 4]
    """
    def flatten(l:Iterable[object]) -> None:
        nonlocal m
        for e in l:
            if isinstance(e, Iterable) and not isinstance(e, str):
                flatten(e)
            else:
                m.append(e)
    m = []
    flatten(l)
    return m

def findDistinct(pattern:str, s:str) -> Iterable[str]:
    """
    Summary:
        Given a potentially long string (s), search for all instance of a pattern and return a set with all distinct occurrences of such pattern.
    Args:
        pattern (str): Regular expression to find.
        s (str): The string where the regular expression is to be found.

    Returns:
        Iterable[str]: An iterable with distinct items.
    """
    return {s.strip() for s in set(flattenList(re.findall(pattern, s, re.IGNORECASE)))}

def fixDiagramNames(diagram_text:str) -> str:
    """
    Summary:
        The text that ChatGPT provides for Sequence Diagram is rife with issues, especially concerning entity names. This function tries to fix most of them.
    Args:
        diagram_text (str): The Diagram Text as returned by ChatGPT

    Returns:
        str: The same Diagram Text, with the (most of the) issues fixed.
    """
    def surroundWith(l:List[str], surround_text:str) -> None:
        nonlocal diagram_text
        for item in l:
            diagram_text = diagram_text.replace(item, f'{surround_text}{item}{surround_text}')
    participant_names = findDistinct('participant|class ([^\r\n{]+)', diagram_text)
    for participant_name in participant_names:
        diagram_text = diagram_text.replace(participant_name, f'"{cleanString(participant_name, "_")}"')

    # Marking all message notations with an &&&, so the regexes later do not get confused.
    message_markers = findDistinct('( *[\\<>\-\*o]*-[\\<>\-\*o]* *)', diagram_text)
    surroundWith(message_markers, '&&&')

    # Wrap all object names with double quotes.
    found_names = findDistinct('(.+)&&&.+&&&(.+) *:.+', diagram_text)
    for found_name in found_names:
        diagram_text = diagram_text.replace(found_name, cleanString(found_name, '_'))

    diagram_text = diagram_text.replace('&&&', ' ')
    diagram_text = diagram_text.replace('--', '-')

    if not participant_names:
        participant_names = '\n'.join([f'participant "{name}"' for name in found_names])
        participant_names = f'@startuml\n\n{participant_names}\n\n'
        diagram_text = diagram_text.replace('@startuml\n', participant_names)

    return diagram_text

def retrieveUMLFromOpenAI(what:str, umlType:str) -> None:
    """
    Summary:
        Add text to current ChatGPT Prompt, then ask ChatGPT for an answer according to umlType, which is a UML diagram type supported by PlantUML. Then place the UML diagram in the UI.
    Args:
        what (str): text to be added to the current ChatGPT prompt.
        umlType (str): A PlantUML type of diagram, such as Class Diagram or Sequence Diagram.
    """
    printG(f'\n\nRetrieving UML {what} diagram...\n')
    st.session_state['PROMPT'] += f'\n\nPlease provide the {what} for this design, using the PlantUML notation.'
    diagram          = queryAIUntilDone(what)
    diagram_position = diagram.rfind('@startuml')
    diagram          = diagram[diagram_position:]
    if umlType == 'Sequence Diagram':
        diagram = fixDiagramNames(diagram)
    placeUMLDiagram(diagram, umlType)

def retrieveTextFromOpenAI(what:str) -> None:
    """
    Summary:
        Add text to current ChatGPT Prompt, then ask ChatGPT for an answer and place the response on the UI.
    Args:
        what (str): text to be added to the current ChatGPT prompt.
    """
    printG(f'\n\nRetrieving {what} using plain text Notation...\n')
    query_to_add = f'\n\nPlease provide the {what} for this system, using plain text Notation.'
    st.session_state['PROMPT'] += (query_to_add + '\n')
    answer = queryAIUntilDone(what)
    st.divider()
    st.subheader(what)
    st.text(answer)

def get_system_design_prompt_sequence() -> List[dict]:
    """
    Summary:
        Retrieves information for all design prompt items.
    Returns:
        List[dict]: A list with all design prompt items, each item with these pieces of information:
            Type: Text or UML. 
                If it is Text, the UI will display the text as it was returned by ChatGPT. 
                If it is UML, this script will convert the text returned by ChatGPT to the notation specified by Notation, see below.
            Notation: Plain Text, Sequence Diagram or Class Diagram.
    """
    return [ 
        { 'Notation' : 'Plain Text',       'What' : 'Use Cases'                                                                                   },
        { 'Notation' : 'Plain Text',       'What' : 'Number of Users, Requests per Day, Average Request Size in Bytes according to the Use Cases' },
        { 'Notation' : 'Plain Text',       'What' : 'Functional Requirements according to the Use Cases'                                          },
        { 'Notation' : 'Plain Text',       'What' : 'Non Functional Requirements according to the Use Cases'                                      },
        { 'Notation' : 'Plain Text',       'What' : 'Schema according to the Use Cases'                                                           },
        { 'Notation' : 'Plain Text',       'What' : 'REST API according to the Use Cases and the Schema'                                          },
        { 'Notation' : 'Sequence Diagram', 'What' : 'Sequence Diagram according to the REST API and the Schema'                                   },
        { 'Notation' : 'Class Diagram',    'What' : 'Class Diagram according to the Schema'                                                       },
    ]

def placeDesignItems(item_notations_regex_patterns_to_place:str) -> None:
    """
    Summary:
        Retrieves the list of items to be placed in the answer, then place them on the screen one by one.
    Args:
        item_notations_regex_patterns_to_place (str): regular expression pattern indicating which item to be placed. For instance, to place all items, use '.*', to place only the diagrams, use '.*Diagram'.
    """
    sequence = [item for item in get_system_design_prompt_sequence() if re.match(item_notations_regex_patterns_to_place,item['Notation'])]
    for item in sequence:
        if item['Notation'] == 'Plain Text':
            retrieveTextFromOpenAI(item['What'])
        elif item['Notation'] in ['Sequence Diagram', 'Class Diagram']:
            retrieveUMLFromOpenAI(item['What'], item['Notation'])
    addReloadButton()  

def go() -> None:
    """
    Starts the solution for the system design question.
    """
    printG('Retrieving system design...')
    st.title(st.session_state['SYSTEM_TO_DESIGN'])
    openai.api_key               = os.environ['OPENAI_API_KEY']
    st.session_state['CACHE_ID'] = cleanString(f"{st.session_state['JOB_TITLE']}_{st.session_state['SYSTEM_TO_DESIGN']}", '_').lower() 
    st.session_state['PROMPT']   = f"Act as if you are a {st.session_state['JOB_TITLE']} being interviewed and are asked to answer this system design question: \n{st.session_state['SYSTEM_TO_DESIGN']}.\n"
    placeDesignItems('.*')
    st.success('Done!')


def initialChecks() -> bool:
    """
    Summary:
        Checks if:
            1. We have the OpenAI API Key.
            2. Java is installed.
            3. If environment variable PLANT_UML_PATH with the path for PlantUML exists.
            4. If the file specified in the environment variable PLANT_UML_PATH exists.
    Returns:
        bool: True if all checks are satisfied. Otherwise, it will return False and it will display an error message describing what needs to be done in the page.

    """
    try:
        _ = os.environ['OPENAI_API_KEY']
    except KeyError:
        st.error("Please create an environment variable OPENAI_API_KEY with the key obtained from https://platform.openai.com/account/api-keys, then restart this environment.")
        return False

    try:
        _ = os.environ['PLANT_UML_PATH']
    except KeyError:
        st.error("Please create an environment variable PLANT_UML_PATH with the path for PlantUML after you downloaded it from https://plantuml.com/download, then restart this environment.")
        return False

    if not os.path.exists(os.environ['PLANT_UML_PATH']):
        st.error(f"Coult not find the file specified in the environment variable PLANT_UML_PATH [{os.environ['PLANT_UML_PATH']}]. Please correct the path for PlantUML, then restart this environment.")
        return False    

    try:
        _ = subprocess.check_output(['java', '--version'])
    except FileNotFoundError:
        st.error("Please install Java https://platform.openai.com/account/api-keys, then restart this environment. We need it to execute PlantUML.")
        return False

    return True
 
if initialChecks():
    st.title("Lulu's System Design Interview Helper")
    st.session_state['JOB_TITLE']        = st.text_input('Job Title', value='Software Development Manager')
    st.session_state['SYSTEM_TO_DESIGN'] = st.text_input('System to be designed:', value='TicketMaster')
    if os.environ['OPENAI_API_KEY'] and st.session_state['JOB_TITLE'] and st.session_state['SYSTEM_TO_DESIGN']:
        st.button(label='GO', on_click=go, key='GO')


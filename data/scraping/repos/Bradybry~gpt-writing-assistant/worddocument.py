from pathlib import Path
from tqdm import tqdm
import time
from lxml import etree
from uuid import uuid4
import re
from zipfile import ZipFile
from diff_match_patch import diff_match_patch
from typing import Dict, List, Tuple

from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

import os
import win32com.client as win32

openai_api_key = os.environ.get('OPENAI_API_KEY')
anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
if openai_api_key is None:
    from dotenv import load_dotenv
    load_dotenv()
    openai_api_key = os.environ.get('OPENAI_API_KEY')

if anthropic_api_key is None:
    from dotenv import load_dotenv
    load_dotenv()
    anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')


class WordEditor:
    def __init__(self, model_params: dict, preamble: str) -> None:

        self.model_params = model_params

        if 'gpt' in model_params["model_name"]:
            if model_params["model_name"] == "gpt-3.5-turbo-16k":
                model_params["max_tokens"] = 6144
            if model_params["model_name"] == "gpt-4":
                model_params["max_tokens"] = 4096
            self.llm = ChatOpenAI(
                openai_api_key=openai_api_key,
                **model_params,
                request_timeout=600
            )
        elif 'claude' in model_params["model_name"]:
            if model_params['model_name'] in ['claude-instant-1-100k', 'claude-1-100k']:
                self.model_params['max_tokens'] = 15000
            if model_params["model_name"] in ["claude-instant-1", "claude-1"]:
                self.model_params["max_tokens"] = 4096
            self.llm = ChatAnthropic(
                model=self.model_params["model_name"],
                temperature=self.model_params["temperature"],
                max_tokens_to_sample=self.model_params["max_tokens"],
                anthropic_api_key=anthropic_api_key
            )

        self.system_message = SystemMessage(content=preamble)

    def edit_paragraphs(self, paragraphs: List[str]):
        num_paragraphs = len(paragraphs)
        prompt = self.get_prompt(paragraphs)
        messages = [self.system_message, prompt]
        llm_output = self.llm(messages).content
        parsed_output = self.parse_llm_output(llm_output, num_paragraphs)

        if parsed_output['parsed']:
            return parsed_output['changes_necessary'], parsed_output['edited_text']
        else:
            return [0] * num_paragraphs, paragraphs

    def get_prompt(self, paragraphs: List[str]):
        prompt = "".join(
            f"<paragraph_{i + 1}>{text}</paragraph_{i + 1}>\n"
            for i, text in enumerate(paragraphs)
        )
        return HumanMessage(content=prompt)
    
    def parse_llm_output(self, llm_output: str, num_paragraphs: int) -> Dict[str, list]:
        """
        Parse the output of the LLM API for tracked changes and return a dictionary of parsed values.
        """
        try:
            edited_text_pattern = r'<edited_p(\d+)>(.*?)<\/edited_p(\d+)>'
            edited_text = re.findall(edited_text_pattern, llm_output, re.DOTALL)

            if len(edited_text) <= 0:
                return {
                    'changes_necessary': [False] * num_paragraphs,
                    'edited_text': [''] * num_paragraphs,
                    'parsed': False
                }
            change_necessary, edited_text_values  = self.extract_changes_and_edited_texts(edited_text, num_paragraphs)

            return {
                'changes_necessary': change_necessary,
                'edited_text': edited_text_values,
                'parsed': True
            }
        except Exception as e:
            return {
                'changes_necessary': [False] * num_paragraphs,
                'edited_text': [''] * num_paragraphs,
                'parsed': False
            }
    def extract_changes_and_edited_texts(self, edited_text: list, num_paragraphs: int) -> Tuple[List[bool], List[str]]:
        """
        Extract the changes and edited texts from the edited text.
        """
        change_necessary = []
        edited_text_values = [e[1].strip() for e in edited_text][:num_paragraphs]
        for text in edited_text_values:
            if text == 'No Change':
                change_necessary.append(False)
            else:
                change_necessary.append(True)
        
        return change_necessary, edited_text_values


class WordDocument:
    def __init__(self, path : Path, author : str = "Unpaid Intern"):
        self.path = Path(path)
        self.author = author
        now = time.time()
        date_struct = time.localtime(now)
        self.date = time.strftime("%Y-%m-%dT%H:%M:%SZ", date_struct)
        self.xml_root = self.extract_xml_root()
        self.nsmap = {
            "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
            "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
            "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
        }
        self.paragraphs = self.get_paragraphs()
        self.log_file = Path(f'./logs/{now}_{self.path.stem}.log')

        if not self.log_file.parent.exists():
            os.makedirs(self.log_file.parent)
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Author: {self.author}\n")
            f.write(f"Date: {self.date}\n")
            f.write(f"Path: {self.path.stem}\n")
            f.write(f"Number of paragraphs: {len(self.paragraphs)}\n")
            f.write(f"Number of words: {len(''.join(self.get_paragraphs_text(self.paragraphs)).split(' '))}\n\n")
    
    def log(self, text: str):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(text)        

    def extract_xml_root(self):
        # Load the document as a ZipFile
        with ZipFile(self.path, "r") as docx_zip:
            # Read the document.xml file into memory
            with docx_zip.open("word/document.xml", "r") as document_xml:
                # Parse the XML content
                xml_content = document_xml.read().decode("utf-8")
                xml_content = re.sub(r'^<\?xml[^>]*\?>', '', xml_content)  # Remove XML encoding declaration if present
                xml_tree = etree.fromstring(xml_content)
                xml_root = xml_tree

        return xml_root

    def get_xml_root(self):
        return self.xml_root
    
    def get_nsmap(self):
        return self.nsmap
    
    def save_tracked_changes_docx(self, output_path: Path):
        """
        Save the modified document with tracked changes.
        
        Args:
            docx_path (Path): The path to the input Word document.
            
        Returns:
            Path: The path to the modified document with tracked changes applied.
        """
        modified_xml = self.serialize_xml()
        with ZipFile(self.path, 'r') as docx_zip:
            # Create a new ZipFile to write the modified document
            with ZipFile(output_path, 'w') as modified_zip:
                for item in docx_zip.infolist():
                    if item.filename == 'word/document.xml':
                        # Replace the document.xml file with the modified XML content
                        modified_zip.writestr(item, modified_xml)
                    else:
                        modified_zip.writestr(item, docx_zip.read(item.filename))

    def serialize_xml(self):
        return etree.tostring(
            self.get_xml_root(),
            pretty_print=True,
            xml_declaration=True,
            encoding='UTF-8',
        )
    
    def apply_tracked_changes_to_paragraph(self, paragraph: etree.Element, original_text: str, edited_text: str):
        unique_id = uuid4()
        nsmap = self.get_nsmap()

        paragraph_properties = paragraph.find(f"./{{{nsmap['w']}}}pPr", namespaces=nsmap)

        self.remove_elements(paragraph)

        diffs = self.compute_diffs(original_text, edited_text)

        if paragraph_properties is not None:
            paragraph.insert(0, paragraph_properties)

        self.reconstruct_paragraph(paragraph, diffs, unique_id)

    def compute_diffs(self, original_text: str, edited_text: str):
        dmp = diff_match_patch()
        diffs = dmp.diff_main(original_text, edited_text)
        dmp.diff_cleanupSemantic(diffs)
        return diffs

    def remove_elements(self, paragraph: etree.Element):
        W_R = f"{{{self.nsmap['w']}}}r"
        for elem in paragraph.findall(f"./{W_R}", namespaces=self.nsmap):
            paragraph.remove(elem)

    def reconstruct_paragraph(self, paragraph: str, diffs: list, unique_id: uuid4):
        for op, text in diffs:
            if op == 0:  # Equality
                self.insert_equal_text(paragraph, text)

            elif op == 1:  # Insertion
                self.insert_new_text(paragraph, text, unique_id)

            elif op == -1:  # Deletion
                self.delete_text(paragraph, text, unique_id)

    def insert_equal_text(self, paragraph: etree.Element, text: str):
        W_NS = self.nsmap["w"]
        W_R = f"{{{W_NS}}}r"
        W_T = f"{{{W_NS}}}t"
        run_element = etree.SubElement(paragraph, W_R)
        text_element = etree.SubElement(run_element, W_T)
        text_element.text = text
        text_element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

    def insert_new_text(self, paragraph: etree.Element, text: str, unique_id: uuid4):
        W_NS = self.nsmap["w"]
        W_R = f"{{{W_NS}}}r"
        W_T = f"{{{W_NS}}}t"
        W_INS = f"{{{W_NS}}}ins"        
        ins_element = etree.SubElement(
            paragraph, W_INS,
            attrib={f"{{{W_NS}}}id": str(unique_id), f"{{{W_NS}}}author": self.author, f"{{{W_NS}}}date": self.date}
        )
        run_element = etree.SubElement(ins_element, W_R)
        text_element = etree.SubElement(run_element, W_T)
        text_element.text = text
        text_element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

    def delete_text(self, paragraph: etree.Element, text: str, unique_id: uuid4):
        W_NS = self.nsmap["w"]
        W_R = f"{{{W_NS}}}r"
        W_DEL = f"{{{W_NS}}}del"
        W_DEL_TEXT = f"{{{W_NS}}}delText"        
        del_element = etree.SubElement(
            paragraph, W_DEL,
            attrib={f"{{{W_NS}}}id": str(unique_id), f"{{{W_NS}}}author": self.author, f"{{{W_NS}}}date": self.date}
        )
        run_element = etree.SubElement(del_element, W_R)
        del_text_element = etree.SubElement(run_element, W_DEL_TEXT)
        del_text_element.text = text
        del_text_element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

    def save_xml(self, output_path : Path):
        xml = self.serialize_xml()
        with open(output_path, "wb") as f:
            f.write(xml)

    def get_paragraphs(self):
        return self.xml_root.xpath(".//w:p[w:r/w:t and not(.//pic:pic) and not(.//wps:wsp)]", namespaces=self.nsmap)

    def get_paragraphs_text(self, paragraphs: etree.Element):
        texts = []
        for p in paragraphs:
            markdown_text = ""
            for run in p.xpath(".//w:r", namespaces=self.nsmap):
                text = ''.join(run.xpath(".//w:t/text()", namespaces=self.nsmap))
                if props_list := run.xpath(".//w:rPr", namespaces=self.nsmap):
                    props = props_list[0]

                    # Check for bold
                    if props.xpath(".//w:b", namespaces=self.nsmap):
                        text = f"**{text}**"

                    # Check for italics
                    if props.xpath(".//w:i", namespaces=self.nsmap):
                        text = f"*{text}*"

                markdown_text += text
            texts.append(markdown_text)
        return texts

    def get_next_n_paragraphs(self, start_idx: int, max_chars: int = 4000):
        total_chars = 0
        end_idx = start_idx
        while end_idx < len(self.paragraphs) and total_chars < max_chars:
            current_text = ''.join(self.paragraphs[end_idx].xpath(".//w:t/text()", namespaces=self.nsmap))
            total_chars += len(current_text)
            if total_chars < max_chars:
                end_idx += 1
            else:
                break
        return self.paragraphs[start_idx:end_idx]
    
    def edit_paragraphs(self, llm : WordEditor):
        """
        Get edited paragraphs using the AI model.

        Args:
            doc: WordDocument object containing the document to be edited.

        Returns:
            None
        """
        model_info = f'Model: {llm.model_params["model_name"]}\n'
        model_info += f'Temperature: {llm.model_params["temperature"]}\n'
        model_info += f'Max Tokens: {llm.model_params["max_tokens"]}\n\n'
        self.log(model_info)
        paragraphs = self.paragraphs
        progress_bar = tqdm(total=len(paragraphs), desc="Processing paragraphs", ncols=100)
        i = 0
        while i < len(paragraphs):
            max_chars = llm.model_params['max_tokens'] * 2
            current_paragraphs = self.get_next_n_paragraphs(i, max_chars=max_chars)

            original_texts = self.get_paragraphs_text(current_paragraphs)
            
            if all(text == '' for text in original_texts):
                stats = [0] * len(current_paragraphs)
                edited_texts = original_texts
            else:
                stats, edited_texts = llm.edit_paragraphs(original_texts)

            for idx, (stat, edited_text) in enumerate(zip(stats, edited_texts)):
                log_string = f"Paragraph {i+idx+1} of {len(paragraphs)}: {stat}\n"
                log_string += f"Original text: {original_texts[idx]}\n"
                log_string += f"Edited text: {edited_text if stat else 'No Change'}\n\n"
                self.log(log_string)
                if stat:
                    paragraph = current_paragraphs[idx]
                    self.apply_tracked_changes_to_paragraph(paragraph, original_texts[idx], edited_text)
            progress_bar.update(len(edited_texts))
            i += len(edited_texts)
        progress_bar.close()

    def run_vba_macro(self, file_path: Path):
        file_path = str(file_path.absolute())
        # Ensure Word is visible
        word = win32.gencache.EnsureDispatch('Word.Application')
        word.Visible = True

        # Open the Word Document
        doc = word.Documents.Open(file_path)

        # Access the VBA project object and run the macro
        word.Application.Run("Normal.Module1.ConvertMarkdownToWordFormat")

        # Save and Close
        doc.Save()
        doc.Close()

        # Quit Word Application
        word.Quit()


def run_doc_review(input_path, output_path, model_params, preamble):
    if type(input_path) == str:
        input_path = Path(input_path)
    if type(output_path) == str:
        output_path = Path(output_path)
    if not output_path.parent.exists():
        os.makedirs(output_path.parent)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} not found.")
    if input_path.suffix != ".docx":
        raise ValueError(f"Input file {input_path} must be a .docx file.")
    if output_path.suffix != ".docx":
        raise ValueError(f"Output file {output_path} must be a .docx file.")

    doc = WordDocument(input_path)
    llm = WordEditor(model_params, preamble)
    doc.edit_paragraphs(llm)
    doc.save_tracked_changes_docx(output_path)
    doc.run_vba_macro(output_path)

if __name__ == "__main__":
    model_params = {
        "model_name": "claude-v1.3",
        "temperature": 0.00,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "n": 1,
        "max_tokens": 4096
    }
    preamble = open('./preamble.txt', 'r').read()

    run_doc_review(
        input_path="./uploads/gpt_jokes.docx",
        output_path="./output/gpt_jokes.docx",
        model_params=model_params,
        preamble=preamble
    )
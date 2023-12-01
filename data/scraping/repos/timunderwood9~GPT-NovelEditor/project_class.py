import json
from tiktoken import encoding_for_model
import math
import openai
import re
import tkinter as tk

class Section:
    def __init__(self, section_text, name, llm_outputs = ""):
        self.name = name
        self.section_text = section_text
        self.llm_outputs = llm_outputs

    def to_dict(self):
        return {
            'name' : self.name,
            'section_text': self.section_text,
            'llm_outputs': self.llm_outputs
        }

class Chapter:
    def __init__(self, name, text):
        self.sections = []
        self.name = name
        self.text = text

    def add_section(self, section):
        self.sections.append(section)

    def to_dict(self):
        return {
            'name' : self.name,
            'text' : self.text,
            'sections': [section.to_dict() for section in self.sections]
        }

class Project:
    def __init__(self, **kwargs):
        self.api_key = ""
        self.title = ""
        self.chapters = []
        self.project_text = ""
        self.current_prompt = ""
        self.gpt4_flag = 0
        self.divided = False
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def add_chapter(self, chapter):
        self.chapters.append(chapter)

    def get_attributes_as_dict(self):
        return {
            'api_key' : self.api_key,
            'title' : self.title,
            'project_text' : self.project_text,
            'current_prompt' : self.current_prompt,
            'gpt4-flag' : self.gpt4_flag,
            'current_prompt' : self.current_prompt,
            'divided' : self.divided,
            'chapters': [chapter.to_dict() for chapter in self.chapters]
        }

    def save(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.get_attributes_as_dict(), file)
    
    @staticmethod
    def load(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            project = Project()
            project.api_key = data['api_key']
            project.title = data['title']
            project.project_text = data['project_text']
            project.current_prompt = data['current_prompt']
            project.gpt4_flag = data['gpt4-flag']
            project.divided = data['divided']
            project.chapters = Project.chapters_from_dict(data['chapters'])
        
        print (project.api_key)
        print (project.chapters)
        return project

    @staticmethod
    def chapters_from_dict(chapter_list):
        chapters = []
        for chapter in chapter_list:
            name = chapter['name']
            text = chapter['text']
            new_chapter = Chapter(name, text)
            section_list = []
            for section in chapter['sections']:
                name = section['name']
                section_text = section['section_text']
                llm_outputs = section['llm_outputs']
                new_section = Section(section_text, name, llm_outputs=llm_outputs)
                section_list.append(new_section)
            new_chapter.sections = section_list
            chapters.append(new_chapter)
        return chapters

    
    def split_text_into_chunks(self, text, max_chunk_size=2000, overlap=40):
        chunks = []
        index = 0
        encoded_text = self.encode_text(text)
        number_of_sections = 1
        if len(encoded_text) > max_chunk_size:
            number_of_sections = math.ceil(len(encoded_text)/(max_chunk_size+overlap))
        chunk_size = math.ceil(len(encoded_text)/ number_of_sections) + overlap

        while index < len(encoded_text):
            decoded_chunk= self.decode_text(encoded_text[index:index+chunk_size])
            chunks.append(decoded_chunk)
            index += chunk_size - overlap

        return chunks

    #This creates a chapter
    def split_chapter(self, chapter):
        chunks = self.split_text_into_chunks(chapter.text)
        i = 1
        for chunk in chunks:
            section_name = f'{chapter.name}: Section {i}'
            section = Section(chunk, section_name)
            chapter.add_section(section)
            i += 1

    def split_all_chapters(self):
        for chapter in self.chapters:
            self.split_chapter(chapter)

    def create_sections_and_chapters_from_text(self, flag, divider = 'Chapter'):
        print(flag)
        if not flag:
            chapter = Chapter(f'{self.title}', self.project_text)
            self.add_chapter(chapter)
        
        else:
            i = 0
            start_index = 0
            for match in re.finditer(divider, self.project_text):
                end_index = match.start()
                if end_index-start_index <= 200:
                    start_index = end_index
                    continue
                chapter_name = f'Chapter {i}'
                chapter = Chapter(chapter_name, self.project_text[start_index:end_index])
                self.add_chapter(chapter)
                i += 1
                start_index = end_index
        
        self.split_all_chapters()

    def create_pdf_of_gpt_outputs(self):
        pdf_text = []
        for chapter in self.chapters:
            for section in chapter.sections:
                heading = self.make_bold(section.name + ':\n')
                self.add_to_pdf(heading)
                text += section.llm_results + '\n'
                self.add_to_pdf(text)
        pdf = self.create_pdf(pdf_text)
        self.open_save_dialogue(pdf)

    def make_bold(self, text):
        pass

    @staticmethod
    def create_project(project_title=None):
        if project_title == None:
            project_title = 'project'
        project = Project()
        project.title = project_title
        return project
    

    def encode_text(self, text):
        return encoding_for_model('gpt-4').encode(text)

    def decode_text(self, text):
        return encoding_for_model('gpt-4').decode(text)
    

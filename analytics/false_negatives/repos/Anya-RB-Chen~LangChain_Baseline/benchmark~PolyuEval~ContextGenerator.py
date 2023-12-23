import os
import re
import fitz
import docx2txt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from InstructorEmbedding import INSTRUCTOR
import torch
from bs4 import BeautifulSoup

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = INSTRUCTOR('../instructor-embedding/instructor-large')
model = INSTRUCTOR('../instructor-embedding/instructor-large')

class ContextGenerator:
    def __init__(self, knowledge_dir = 'data', k = 5, template = None):
        self.knowledge_dir = knowledge_dir  # Path to the knowledge base
        self.template = template
        if template:
            self.knowledge_dir = os.path.join(self.knowledge_dir, template)
        self.k = k  # The number of most related paragraphs to be included in the prompt
        self._read_paragraphs()

    # Read all pdf, docx, txt and html files in the given directory and convert them into plain text
    def _read_files(self, directory):
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.txt'):
                    filepath = os.path.join(root, filename)

                    with open(filepath, 'rb') as f:
                        if filename.endswith('.txt'):
                            # encode with utf-8 to avoid UnicodeDecodeError
                            text = f.read().decode('utf-8')
                        yield text

    # Split the text into chunks with a length of chunk_length and clean their format
    def _split_text(self, text):
        if self.template == 'A' or self.template == 'B':
            text_chunks = text.split('\n')
        elif self.template == 'C':
            text_chunks = []
            temp = ''
            for i, line in enumerate(text.split('\n')):
                if i % 4 == 0 and i != 0:
                    text_chunks.append(temp)
                    temp = ''
                temp += line
                temp += '\n'
            text_chunks.append(temp)
        else:
            raise ValueError("The template should be A, B or C.")
        return text_chunks

    # Paragraph embedding
    def _embed_paragraphs(self, paragraphs, chunk_length):
        text_instruction_pairs = []
        
        if type(paragraphs) == str:
            text_instruction_pairs.append([" ", paragraphs])
        elif type(paragraphs) == list:
            for paragraph in paragraphs:
                text_instruction_pairs.append([" ", paragraph])
        else:
            raise TypeError("The type of paragraphs should be str or list.")
        # print(len(text_instruction_pairs))
        # print(text_instruction_pairs[0])
        customized_embeddings = model.encode(text_instruction_pairs)
        return customized_embeddings

    # Calculate the similarity between the question embedding and the embedding of each paragraph,
    # and find the k paragraphs with the highest similarity
    def _find_similar_paragraphs(self, embedded_paragraphs, embedded_question, k):
        
        similarity_scores = cosine_similarity(embedded_paragraphs, embedded_question)
        top_k_indices = np.argsort(similarity_scores, axis=0)[-k:].flatten()
        top_k_scores = similarity_scores[top_k_indices].flatten()
        return top_k_indices, top_k_scores

    # Prompt generation
    def _generate_prmopt(self, paragraphs, top_k_indices, question, top_k_scores):
        context = ''
        # rank the paragraphs according to the similarity score from high to low
        for i in range(len(top_k_indices)):
            context += paragraphs[top_k_indices[i]]
            context += '\n'
        # print(context)
        return context
    
    def _read_paragraphs(self):
        self._text_data = list(self._read_files(self.knowledge_dir))
        self._paragraphs = []
        for text in self._text_data:
            self._paragraphs += self._split_text(text, self.chunk_length)
        # print(self._paragraphs)

        self._embedded_paragraphs = self._embed_paragraphs(self._paragraphs, self.chunk_length)
    

    def get_top_k(self, question):
        embedded_question = self._embed_paragraphs(question, self.chunk_length)
        top_k_indices, top_k_scores = self._find_similar_paragraphs(self._embedded_paragraphs,
                                                embedded_question, self.k)
        
        context = self._generate_prmopt(self._paragraphs, top_k_indices, question, top_k_scores)
        return context


if __name__ == '__main__':
    cg = ContextGenerator(template='A')
    context = cg.get_top_k("What is the course code of the course 'Computer Vision'?")
    print(context)

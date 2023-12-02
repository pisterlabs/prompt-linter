import re
import cohere
import streamlit as st
import os


# Write a class to combine the functions above, and add a function to generate a new question from the context
class QuestionGenerator:
    string = """
            Question: 
            Which of the following was not the cause of the 1918 flu pandemic?
            Options:
            A H1N1 influenza A virus.
            B H2N2 influenza A virus.
            C H3N3 influenza A virus.
            D H5N1 influenza A virus.
            Answer:
            D
            """
    def __init__(self, context):
        self.context = context
        self.co = cohere.Client(os.getenv('COHERE_API_KEY'))

    def generate_question(self):
        response = self.co.generate(
          model='command',
          prompt = "Based on the context, write a multiple choice question with 4 options.  Use the format:\nQuestion:(Question here) \nOptions:\n((A) OPTION 1 here) \n((B) OPTION 2 here) \n((C) OPTION 3 here), \n((D) OPTION 4 here) \nanswer: (CORRECT OPTION here)\n\nContext: \'{}\'".format(self.context),

        #   prompt=f'Instructions: Write a multiple choice question with 4 choices from a given context\nContext\"{self.context}\"\nOutput format should be\n[question:QUESTION, options:[OPTION 1, OPTION 2, OPTION 3, OPTION 4], answer: CORRECT OPTION]\n',
          max_tokens=2698,
          temperature=0.9,
          k=0,
          stop_sequences=[],
          return_likelihoods='NONE')
        string = response.generations[0].text
        print(string)
        dictionary = self.string_dict(string)
        return self.test_string_dict(dictionary)
    
    def generate_quest(self):
        print(self.context)
        response = self.co.generate(
        model='command',
        prompt = "Write a question and a short answer based on the given context. Use the format:\n\nQUESTION: (Your question here)\nANSWER: (Your answer here)\n\nCONTEXT: \'{}\'".format(self.context),
        # prompt=f'Generate a question and a short phrase answer, from a given context.\n\nCONTEXT: \"{self.context}\"\nOUTPUT should return:\nQUESTION:{{QUESTION}}\nANSWER: {{ANSWER}}',
        max_tokens=300,
        temperature=0.9,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE')
        string = response.generations[0].text
        print(string)
        dictionary = self.str_dict(string)
        return self.test_str_dict(dictionary)
    
    def string_dict(self, str_):
        pattern = re.compile(r'Question:(.*)Options:(.*)Answer:(.*)', re.DOTALL | re.IGNORECASE)
        match = pattern.search(str_)
        try:
            question = match.group(1).strip()
            options = [option.strip() for option in match.group(2).split('\n')]
            answer = match.group(3).strip()
            dictionary = {'question': question, 'options': options, 'answer': answer}
            dictionary['options'] = list(filter(None, dictionary['options']))

            dictionary['answer'] = answer[answer.find("(")+1:answer.find(")")]
            dictionary['options'] = {option[:4].strip().replace(")","").replace("(",""): option[4:] for option in dictionary['options']}
            return dictionary
        except:
            question = None
            options = None
            answer = None

            return False


    
    def str_dict(self, str_):
        pattern = re.compile(r"QUESTION:\s*(.+)\nANSWER:\s*(.+)")
        match = pattern.search(str_)
        try:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            dictionary = {'question': question, 'answer': answer}
            return dictionary
        except:
            st.write("Try Again")
            question = None
            answer = None
            return False
            

    
    def test_string_dict(self, dictionary):
        try:
            assert 'question' in dictionary.keys()
            assert 'options' in dictionary.keys()
            assert 'answer' in dictionary.keys()
            assert len(dictionary['options']) == 4
            assert dictionary['answer'] in dictionary['options'].keys()
            assert all(dictionary['options'].values())

            return dictionary
        except:
            return False

    def test_str_dict(self, dictionary):
        try:
            assert 'question' in dictionary.keys()
            assert 'answer' in dictionary.keys()
            return dictionary
        except:
            return False

class Rephraser:
    """Given a paragraph, rephrase or summarize or correct vocabulary using cohere API"""
    def __init__(self, paragraph):
        self.paragraph = paragraph
        self.co = cohere.Client(os.getenv('COHERE_API_KEY'))
    
    def summarize(self):
        response = self.co.summarize( 
                        text=f'{self.paragraph}',
                        length='medium',
                        format='paragraph',
                        model='summarize-xlarge',
                        additional_command='',
                        temperature=0.3,
                        ) 
        return response.summary
    def reword(self):
        # set the prompt for paraphrasing
        prompt = f"Rephrase this sentence in a different way: {self.paragraph}"

        # generate a response using the multilingual-22-12 model
        response = self.co.generate(
            model="command-nightly",
            prompt=prompt,
            max_tokens=1000,

        )
        # get the generated text
        rephrased_text = response[0].text
        return rephrased_text

    def correct(self):
        prompt = f"Correct this sentence grammar, if there is no error return the same text: {self.paragraph}"

        # generate a response using the multilingual-22-12 model
        response = self.co.generate(
            model="command-nightly",
            prompt=prompt,
            max_tokens=1000,

        )
        # get the generated text
        rephrased_text = response[0].text
        return rephrased_text
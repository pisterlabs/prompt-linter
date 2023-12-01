import pdfplumber
from Levenshtein import distance
import openai


class PDFDatesFinder:
    def __init__(self, path, API_KEY):
        self.path = path
        openai.api_key = API_KEY

    def _remove_noise_from_pages(self, pages):
        """
        Remove noisy lines from pages.
        """
        pages_lines = [page.split('\n') for page in pages]
        pages_lines = [[element.strip() for element in sublist] for sublist in pages_lines]
        
        duplicate_lines = set()

        for i in range(len(pages) - 1):

            for line in pages_lines[i]:
                
                for next_line in pages_lines[i + 1]:
                    
                    if distance(line, next_line) <= 1:
                        duplicate_lines.add(line)

        cleaned_pages = []
        for page in pages_lines:
            cleaned_page = [line for line in page if line not in duplicate_lines]
            cleaned_pages.append(cleaned_page)

        cleaned_pages = ["\n".join(page) for page in cleaned_pages]

        return cleaned_pages
    
    # ChatGPT-3.5
    def _ChatGPT3_conversation(self, conversation):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-1106',
            messages=conversation
        )
        conversation.append({'role': response.choices[0].message.role, 
                            'content': response.choices[0].message.content})
        return conversation

    def extract_text(self):
        """
        Extracts cleaned text in pdf file.

        """
        with pdfplumber.open(self.path) as pdf:
            len_doc = len(pdf.pages)
            pages = []
            for i in range(len_doc):
                pages.append(pdf.pages[i].extract_text(layout=True))

            pages = self._remove_noise_from_pages(pages)
        return pages

    
    def identify_period_pages(self, pages):
        """
        text_pages_dates: list of pages that might contain the policy period -> GPT-4
        index_pages_dates: list of indexes of pages that might contain the policy period -> GPT-4V
        """
        
        text_pages_dates = []
        index_pages_dates = []
        
        count = 1
        for page in pages:
            page = " ".join(page.split())

            conversation = []

            prompt = f"""\
            You are tasked with finding dates in the text.

            If dates ARE found, output in the following format:
            ```
            YES
            ```
            If dates ARE NOT found, output the following:
            ```
            NO
            ```

            Find dates from the following text:

            ```
            {page}
            ```\
            """
            conversation.append({'role': 'user', 'content': prompt})
            conversation = self._ChatGPT3_conversation(conversation)

            response = conversation[-1]['content'].strip()

            if response=='YES':
                text_pages_dates.append(page)
                index_pages_dates.append(count)
            

            count += 1
        
        return text_pages_dates, index_pages_dates
    


class PDFDeductiblesFinder:
    def __init__(self, path, API_KEY):
        self.path = path
        openai.api_key = API_KEY

    def _remove_noise_from_pages(self, pages):
        """
        Remove noisy lines from pages.
        """
        pages_lines = [page.split('\n') for page in pages]
        pages_lines = [[element.strip() for element in sublist] for sublist in pages_lines]
        
        duplicate_lines = set()

        for i in range(len(pages) - 1):

            for line in pages_lines[i]:
                
                for next_line in pages_lines[i + 1]:
                    
                    if distance(line, next_line) <= 1:
                        duplicate_lines.add(line)

        cleaned_pages = []
        for page in pages_lines:
            cleaned_page = [line for line in page if line not in duplicate_lines]
            cleaned_pages.append(cleaned_page)

        cleaned_pages = ["\n".join(page) for page in cleaned_pages]

        return cleaned_pages
    
    # ChatGPT-3.5
    def _ChatGPT3_conversation(self, conversation):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-1106',
            messages=conversation
        )
        conversation.append({'role': response.choices[0].message.role, 
                            'content': response.choices[0].message.content})
        return conversation

    def extract_text(self):
        """
        Extracts cleaned text in pdf file.

        """
        with pdfplumber.open(self.path) as pdf:
            len_doc = len(pdf.pages)
            pages = []
            for i in range(len_doc):
                pages.append(pdf.pages[i].extract_text(layout=True))

            pages = self._remove_noise_from_pages(pages)
        return pages

    
    def identify_deductibles_pages(self, pages):
        """
        text_pages_dates: list of pages that might contain the policy period -> GPT-4
        index_pages_dates: list of indexes of pages that might contain the policy period -> GPT-4V
        """
        
        text_pages_deductibles = []
        index_pages_deductibles = []
        
        count = 1
        for page in pages:
            page = " ".join(page.split())

            conversation = []

            prompt = f"""\
            You are tasked with finding deductibles in the text.

            If deductibles ARE found, output in the following format:
            ```
            YES
            ```
            If deductibles ARE NOT found, output the following:
            ```
            NO
            ```

            Find deductibles from the following text:

            ```
            {page}
            ```\
            """
            conversation.append({'role': 'user', 'content': prompt})
            conversation = self._ChatGPT3_conversation(conversation)

            response = conversation[-1]['content'].strip()

            if response=='YES':
                text_pages_deductibles.append(page)
                index_pages_deductibles.append(count)
            

            count += 1
        
        return text_pages_deductibles, index_pages_deductibles
    


    



# %%
import os
import io
import time
import PyPDF2
import requests
from bs4 import BeautifulSoup
from datetime import datetime

import pandas as pd
import numpy as np
import nltk
import ast # for string to list conversion
# from scipy import spatial

import openai
# from openai.embeddings_utils import cosine_similarity

import fpdf
import pdfkit
from PyPDF2 import PdfReader, PdfWriter
# import csv

from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

from PIL import Image
import matplotlib.pyplot as plt
# from reportlab.lib.pagesizes import letter
# from reportlab.lib import colors
# from reportlab.pdfgen import canvas
# nltk.download('punkt')

# %%
class AutomatedContentGenerator:
    def __init__(self, api_key, cx, query, num_results, openai_api_key):
        # Initialize any necessary variables, data structures, or connections.
        # This could include a web scraper, data parser, text generator, etc.
        self.api_key = api_key
        self.cx = cx
        self.query = query
        self.num_results = num_results
        self.results = []
        openai.api_key = openai_api_key  # replace with your OpenAI API key
        self.folder_path = f'{self.query}_output'
        os.makedirs(self.folder_path, exist_ok=True)
        self.visited_urls = set()

    ## Step 1: Search Google
    # Web Scraping and Parsing
    # TODO insert prompt, show the results by table
    def search_and_scrape_web(self):
        # Use a web scraper to find and retrieve relevant information from the web based on the query.
        # This could involve a search engine API or scraping specific sites.
        url = 'https://www.googleapis.com/customsearch/v1'
        params = {
            'key': self.api_key,
            'cx': self.cx,
            'q': self.query,
            'num': self.num_results,
            # 'siteSearch': 'https://news.google.com/',
        }
        response = requests.get(url, params=params)
        data = response.json()
        if 'items' in data:
            for item in data['items']:
                self.results.append({
                    'title': item['title'],
                    'link': item['link'],
                    'snippet': item['snippet'],
                    'date': item.get('pagemap', {}).get('metatags', [{}])[0].get('og:updated_time')
                })
    
    def save_search_results_to_excel(self, filename=None):
        if filename is None:
            filename = f'{self.query}.xlsx'
        df = pd.DataFrame(self.results)
        df.to_excel(filename, index=False)
        
    ## Step 2: Scrape web pages and save them in pdf format
    # Using Beautifulsoup and pdfkit
    # TODO show bs result and prepare pdf to be able to download
    def extract_content(self, url):
        # Send a GET request to the webpage
        response = requests.get(url)
        
        # Manually set the encoding to 'UTF-8'
        response.encoding = 'UTF-8'

        # Parse the HTML content of the webpage with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the title of the webpage
        title = f"Title:\n{soup.h1}" if soup.h1 else "No Title"

        # Find the main article content by inspecting the HTML structure
        article = soup.find('main')
        if article is None:
            possible_classes = ['main', 'main-content', 'article_main', 'container', 'head', 'headline', 'panel']
            for class_name in possible_classes:
                article = soup.find('main', class_= class_name)
                if article is not None:
                    break

        # If the main article content still cannot be found, return an empty string
        if article is None:
            print(f"Could not find main content in {url}")
            return ""

        # Extract the text within h2, h3, h4, h5, h6, and p tags
        headers_and_paragraphs = article.find_all(['h2', 'h3', 'h4', 'h5', 'h6', 'p'])

        # Join the text from all tags into a single string with HTML tags
        processed_main_content = ''.join(str(tag) for tag in headers_and_paragraphs)
        
        # Title and the content
        processed_content = f"<h1>{title}</h1>" + processed_main_content
        
        # Limit the content to the first 3000 tokens  
        content = self.limit_content_by_tokens(processed_content, 3000)
        
        return content
    
    # limit the content by tokens instead of words
    def limit_content_by_tokens(self, content, max_tokens):
        tokens = content.split()
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return ' '.join(tokens)
  
    # # limit the content by tokens instead of words
    # def limit_content_by_tokens(self, content, max_tokens):
    #     tokens = []
    #     token_count = 0

    #     for word in content.split():
    #         word_tokens = len(word.split()) + 1  # Add 1 for the space that follows each word
    #         if token_count + word_tokens > max_tokens:
    #             break
    #         tokens.append(word)
    #         token_count += word_tokens

    #     return ' '.join(tokens)

    # save the content to pdf
    def save_content_to_pdf(self, url, content, filename):
        # Add the title and the link at the beginning of the content
        content_with_link = f"{content}\n\n <h3>URL:</h3>\n<p></p>\n<p><a href='{url}'>{url}</a></p>"

        # # Split the content into words
        # words = content_with_title_and_link.split()

        # # Limit the content to the first 4000 words
        # limited_content = ' '.join(words[:4000])

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        options = {
            'no-stop-slow-scripts': True,
            'load-error-handling': 'ignore',
            'encoding': "UTF-8", ### This is important to solve quatation mark problem            
        }
        
        path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
        config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
        
        # # Encode the HTML content as UTF-8 before passing it to pdfkit
        # content_with_keywords_link = content_with_keywords_link.encode('utf-8')
    
        pdfkit.from_string(content_with_link, os.path.join(self.folder_path, filename), configuration=config, options=options) 

    # Loop through the URLs and scrape the content    
    def scrape_and_save(self):
        df = pd.read_excel(f'{self.query}.xlsx')
           
        for index, row in df.iterrows():
            url = row['link']
            # title = row['title']
            if url not in self.visited_urls:
                content = self.extract_content(url)
                pdf_filename = f"web_page_{index}.pdf"
                self.save_content_to_pdf(url, content, pdf_filename)
                self.visited_urls.add(url)
            else:
                print(f"Skipping already visited URL: {url}")
                
            # Sleep for 1 second
            time.sleep(1)
                
    # TODO combine with LangChain.DocumentLoader?
    # TODO as a side project, get pdf files, give summarized and customized output
    def extract_text_from_pdf(self, pdf_path):
        #pdf_file_obj = open(pdf_path, 'rb')
        pdf_reader = PdfReader(pdf_path)
        pdf_content = ''
        
        # for page_num in range(pdf_reader.numPages):
        #     page_obj = pdf_reader.getPage(page_num)
        #     pdf_content += page_obj.extractText()
        # pdf_file_obj.close()
        
        for page in pdf_reader.pages:
            pdf_content += page.extract_text()
        
        return pdf_content

    def save_sentences_to_csv(self, pdf_content):
        sentences = nltk.sent_tokenize(pdf_content)
        df_sentences = pd.DataFrame(sentences, columns=['text'])
        return df_sentences
    
    def get_embedding(self, text, model):
        text = text.replace("\n", " ")
        if model == 'default_model':
            # use default model to get embedding
            pass
        elif model == 'text-embedding-ada-002':
            # use 'text-embedding-ada-002' model to get embedding
            pass
        else:
            raise ValueError(f"Unknown model: {model}")
        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

    # TODO replace to LangChain.Embeddings and LangChain.Vectorstore also FAISS?
    def get_embeddings(self, df_sentences):
        df_sentences['embeddings'] = df_sentences['text'].apply(lambda x: self.get_embedding(x, model='text-embedding-ada-002'))
        return df_sentences

    # def get_similarities(self, df_sentences):
    #     df_sentences['similarities'] = df_sentences['embeddings'].apply(lambda x: cosine_similarity(x, df_sentences['embeddings'].tolist()))
    #     return df_sentences
    
    # TODO show similarities as right sidebar
    def get_similarities(self, df_sentences):
        embeddings = np.stack(df_sentences['embeddings'].values)
        similarities = cs(embeddings)
        df_sentences['similarities'] = list(similarities)
        return df_sentences

    # # What algorithm should we use to cluster the sentences?
    # def create_clustering_image(self, df_sentences):
    #     matrix = np.vstack(df_sentences['embeddings'].values)
    #     n_clusters = 3
    #     kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    #     kmeans.fit(matrix)
    #     df_sentences['Cluster'] = kmeans.labels_
        
    #     # Ensure perplexity is less than the number of samples
    #     perplexity = min(15, len(df_sentences) - 1)
        
    #     tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="random", learning_rate=200)
    #     vis_dims2 = tsne.fit_transform(matrix)
    #     x = [x for x, y in vis_dims2]
    #     y = [y for x, y in vis_dims2]
    #     fig, ax = plt.subplots()
    #     for category, color in enumerate(["purple", "green", "red", "blue"]):
    #         xs = np.array(x)[df_sentences.Cluster == category]
    #         ys = np.array(y)[df_sentences.Cluster == category]
    #         ax.scatter(xs, ys, color=color, alpha=0.3)
    #         avg_x = xs.mean()
    #         avg_y = ys.mean()
    #         ax.scatter(avg_x, avg_y, marker="x", color=color, s=100)
    #     ax.set_title("Clusters identified visualized in language 2d using t-SNE")
    #     fig.canvas.draw()
        
    #     # Create a BytesIO object and save the figure to it in PNG format
    #     buf = io.BytesIO()
    #     fig.savefig(buf, format='png')
    #     buf.seek(0)

    #     # Close the figure
    #     plt.close(fig)

    #     return buf
    
    # TODO show clustering image as left sidebar
    def create_clustering_image(self, df_sentences):
        if len(df_sentences) < 2:
            print("Not enough sentences to cluster")
            pass
        else:
            matrix = np.vstack(df_sentences['embeddings'].values)
            n_clusters = 3
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm.fit(matrix)
            df_sentences['Cluster'] = gmm.predict(matrix)
            
            # Ensure perplexity is less than the number of samples
            perplexity = min(15, len(df_sentences) - 1)
            
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="random", learning_rate=200)
            vis_dims2 = tsne.fit_transform(matrix)
            x = [x for x, y in vis_dims2]
            y = [y for x, y in vis_dims2]
            fig, ax = plt.subplots()
            for category, color in enumerate(["purple", "green", "red"]):
                xs = np.array(x)[df_sentences.Cluster == category]
                ys = np.array(y)[df_sentences.Cluster == category]
                if len(xs) > 0 and len(ys) > 0:  # Check that the arrays are not empty
                    ax.scatter(xs, ys, color=color, alpha=0.3)
                    avg_x = xs.mean()
                    avg_y = ys.mean()
                    ax.scatter(avg_x, avg_y, marker="x", color=color, s=100)
            ax.set_title("Clusters identified visualized in language 2d using t-SNE")
            fig.canvas.draw()
            
            # Create a BytesIO object and save the figure to it in PNG format
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)

            # Close the figure
            plt.close(fig)

            return buf

    # def add_image_to_pdf(self, pdf_path, image_buf):
    #     pdf_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pdf')]
        
    #     pass
    
    # def process_openai_embeddings_csv(self):
    #     # Get a list of all PDF files in the folder
    #     pdf_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pdf')]
    #     for pdf_file in pdf_files:
    #         pdf_path = os.path.join(self.folder_path, pdf_file)
    #         pdf_content = self.extract_text_from_pdf(pdf_path)
    #         df_sentences = self.save_sentences_to_csv(pdf_content)
    #         df_sentences = self.get_embeddings(df_sentences)
    #         df_sentences = self.get_similarities(df_sentences)
    #         df_sentences.to_csv(os.path.join(self.folder_path, f'{self.query}_openai.csv'), index=False)

    #         # Sleep for 1 second
    #         time.sleep(1)
    
    def process_pdfs_and_save_embeddings(self):
        # Get a list of all PDF files in the folder
        pdf_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.folder_path, pdf_file)
            pdf_content = self.extract_text_from_pdf(pdf_path)
            df_sentences = self.save_sentences_to_csv(pdf_content)
            df_sentences = self.get_embeddings(df_sentences)
            df_sentences = self.get_similarities(df_sentences)
            df_sentences.to_csv(os.path.join(self.folder_path, f'{pdf_file}_embeddings.csv'), index=False)
            
            # Sleep for 1 second
            time.sleep(1)
    
    def create_clustering_images_from_saved_embeddings(self):
        # Get a list of all CSV files in the folder
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('_embeddings.csv')]
        for csv_file in csv_files:
            csv_path = os.path.join(self.folder_path, csv_file)
            df_sentences = pd.read_csv(csv_path)
            
            # Convert the embeddings from strings back to lists of floats
            df_sentences['embeddings'] = df_sentences['embeddings'].apply(ast.literal_eval).apply(lambda x: [float(i) for i in x])
            
            # Create the clustering image
            image_buf = self.create_clustering_image(df_sentences)
            
            if image_buf is not None:
                # Convert the image buffer to a PIL image
                image = Image.open(image_buf)
                
                # Convert image to RGB if it's RGBA
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                
                # Save the image as a JPEG file
                image.save(os.path.join(self.folder_path, f'{csv_file}_cluster.jpeg'))
                
                # Sleep for 1 second
                time.sleep(1)
            
    # TODO Show keywords right below the output and then summary followed
    # TODO Increase chuck limit with gpt-3.5-turbo-0613
    def extract_keywords(self, pdf_content):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Extract main 10 keywords from the following text as #Keywords form:\n\n{pdf_content}\n\n tl;dr:",
            temperature=0.5,
            max_tokens=300,
            top_p=1.0,
            frequency_penalty=0.8,
            presence_penalty=0.0
        )
        keywords = response.choices[0].text.strip().split(', ')
        return keywords         
    
    def summarize_text(self, pdf_content):
        # Split the text into chunks of 1000 tokens each
        tokens = pdf_content.split()
        chunks = [' '.join(tokens[i:i + 1500]) for i in range(0, len(tokens), 1500)]

        summaries = []
        for chunk in chunks:
            # Summarize each chunk using the OpenAI API
            prompt = f"Please provide a detailed summary of the following text with keywords max 500 words:\n{chunk}\n\n tl;dr:"
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=2000,
                n=1,
                stop=None,
                temperature=0.5,
            )
            summaries.append(response.choices[0].text.strip())

        # Aggregate the summaries
        aggregated_summary = ' '.join(summaries)

        return aggregated_summary
           
    # def summarize_text(self, pdf_content):
    #     # Split the text into chunks of 2000 tokens each
    #     tokens = pdf_content.split()
    #     # chunks = [' '.join(tokens[i:i + 1000]) for i in range(0, len(tokens), 1000)]

    #     # summaries = []
    #     # for chunk in chunks:
    #     #     # Summarize each chunk using the OpenAI API
    #     #     prompt = f"Please provide a bit detailed summary of the following text with keywords max 300 words:\n{chunk}\n\n tl;dr:"
    #     #     response = openai.Completion.create(
    #     #         model="text-davinci-003",
    #     #         prompt=prompt,
    #     #         max_tokens=1000,
    #     #         n=1,
    #     #         stop=None,
    #     #         temperature=0.5,
    #     #     )
    #     #     summaries.append(response.choices[0].text.strip())

    #     # # Aggregate the summaries
    #     # aggregated_summary = ' '.join(summaries)

    #     # Summarize the aggregated summary
    #     prompt = f"The following text is scrapped text of a webpage. Summarize the text and produce an summary ariticle like Bloomberg or Harvard Business Review, max 600 words with core keywords:\n{tokens}\n\n tl;dr:"
    #     response = openai.Completion.create(
    #         model="text-davinci-003",
    #         prompt=prompt,
    #         max_tokens=3000,
    #         n=1,
    #         stop=None,
    #         temperature=0.5,
    #     )
    #     summary = response.choices[0].text.strip()

    #     return summary
    
    # def save_summary_and_keywords_to_pdf(self, url, title, keywords, summary):
    #     content = f"<h1>{title}</h1>\n\n<h2>Keywords:</h2>\n<p>{', '.join(keywords)}</p>\n\n<h2>Summary:</h2>\n<p>{summary}</p>\n\n<h3>URL:</h3>\n<p><a href='{url}'>{url}</a></p>"
    #     if not os.path.exists(self.folder_path):
    #         os.makedirs(self.folder_path)
    #     options = {
    #         'no-stop-slow-scripts': True,
    #         'load-error-handling': 'ignore',
    #         'encoding': "UTF-8",
    #     }
    #     path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    #     config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
    #     pdfkit.from_string(content, os.path.join(self.folder_path, 'output.pdf'), configuration=config, options=options) 

    # TODO extract output as pdf download or each companies server for next usage using LangChain.Memory?
    def save_summary_and_keywords_to_pdf(self, url, title, keywords, summary):
        content = f"<h1>{title}</h1>\n\n<h2>Keywords:</h2>\n<p>{', '.join(keywords)}</p>\n\n<h2>Summary:</h2>\n<p>{summary}</p>\n\n<h3>URL:</h3>\n<p><a href='{url}'>{url}</a></p>"
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        options = {
            'no-stop-slow-scripts': True,
            'load-error-handling': 'ignore',
            'encoding': "UTF-8",
        }
        path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
        config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
        pdfkit.from_string(content, os.path.join(self.folder_path, 'output.pdf'), configuration=config, options=options)

        # If the summary PDF already exists, append the new summary to it
        pdf_filename = f"{self.query}_summary.pdf"
        pdf_path = os.path.join(self.folder_path, pdf_filename)
        if os.path.exists(pdf_path):
            writer = PdfWriter()

            # Existing PDF
            existing_pdf = PdfReader(open(pdf_path, "rb"))
            for page in existing_pdf.pages:
                writer.add_page(page)

            # New PDF
            new_pdf = PdfReader(open(os.path.join(self.folder_path, 'output.pdf'), "rb"))
            writer.add_page(new_pdf.pages[0])

            # Write the output PDF
            with open(pdf_path, "wb") as f:
                writer.write(f)
        else:
            # If the summary PDF doesn't exist, create a new one
            os.rename(os.path.join(self.folder_path, 'output.pdf'), pdf_path)
    
    def process_saved_pdfs(self):
        df = pd.read_excel(f'{self.query}.xlsx')
        for index, row in df.iterrows():
            url = row['link']
            pdf_filename = f"web_page_{index}.pdf"
            pdf_path = os.path.join(self.folder_path, pdf_filename)
            pdf_content = self.extract_text_from_pdf(pdf_path)
            keywords = self.extract_keywords(pdf_content)

            # Split the PDF content into chunks of 2000 tokens each
            tokens = pdf_content.split()
            chunks = [' '.join(tokens[i:i + 1000]) for i in range(0, len(tokens), 1000)]

            # Summarize each chunk separately
            summaries = []
            for chunk in chunks:
                summary = self.summarize_text(chunk)
                summaries.append(summary)

            # Aggregate the summaries
            aggregated_summary = ' '.join(summaries)
            
            final_summary = self.summarize_text(aggregated_summary)
            
            # Extract the title from the PDF content
            # title_start = pdf_content.find('Title:') + len('Title:')
            # title_end = pdf_content.find('\n', title_start)
            # title = pdf_content[title_start:title_end].strip()

            # Save the summary and keywords to a PDF
            self.save_summary_and_keywords_to_pdf(url, row['title'], keywords, final_summary)

            # Sleep for 1 second
            time.sleep(1)
            
    # # TODO Insert request prompt for article generating process or using OpenAI.FunctionCall?
    # def generate_article(self):
    #     pdf_filename = f"{self.query}_summary.pdf"
    #     pdf_path = os.path.join(self.folder_path, pdf_filename)
    #     # news_title = input("Please enter the title of the news: ")
        
    #     # Get the current date
    #     today = datetime.now().date()
        
    #     if os.path.exists(pdf_path):
    #         pdf_content = self.extract_text_from_pdf(pdf_path)
            
    #         # # Split the text into chunks of 2000 tokens each
    #         # tokens = pdf_content.split()
    #         # chunks = [' '.join(tokens[i:i + 2000]) for i in range(0, len(tokens), 2000)]
            
    #         # Produce an professional and detailed journalism article like Harvard Business Review by using resources below, around 2000 tokens length. Also do paragraphing and extract core keywords from the article created.
    #         prompt = f"The following text is scrapped text from multiple webpages as the results of {self.query}. \n Produce a total professional {self.query} article from the text below around 1000 words:\n{pdf_content}\n"
    #         response = openai.Completion.create(
    #                 model="text-davinci-003",
    #                 prompt=prompt,
    #                 max_tokens=2000,
    #                 n=1,
    #                 stop=None,
    #                 temperature=0.5,
    #             )

    #         # summaries = []
    #         # for chunk in chunks:
    #         #     # Summarize each chunk using the OpenAI API
    #         #     prompt = f"The following text is scrapped text of a webpage. Summarize the text and produce an summary article like Bloomberg or Harvard Business Review, max 600 words with core keywords:\n{chunk}\n\n tl;dr:"
    #         #     response = openai.Completion.create(
    #         #         model="text-davinci-003",
    #         #         prompt=prompt,
    #         #         max_tokens=1000,
    #         #         n=1,
    #         #         stop=None,
    #         #         temperature=0.5,
    #         #     )
    #             # summaries.append(response.choices[0].text.strip())

    #         # # Aggregate the summaries
    #         # generated_article = ' '.join(summaries)
            
    #         generated_article = response.choices[0].text.strip()
            
    #         news_article = f"<h1>Title: The Total Summary of {self.query} </h1> \n <h4> Date: {today} </h4> \n\n <p>{generated_article}<p>"
            
    #         options = {
    #             'no-stop-slow-scripts': True,
    #             'load-error-handling': 'ignore',
    #             'encoding': "UTF-8",
    #         }
    #         path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    #         config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
    #         pdfkit.from_string(news_article, os.path.join(self.folder_path, f"{self.query}_final.pdf"), configuration=config, options=options)

    def generate_article(self):
        pdf_filename = f"{self.query}_summary.pdf"
        pdf_path = os.path.join(self.folder_path, pdf_filename)
        
        # Get the current date
        today = datetime.now().date()
        
        if os.path.exists(pdf_path):
            pdf_content = self.extract_text_from_pdf(pdf_path)
            
            # Produce an professional and detailed journalism article like Harvard Business Review by using resources below, around 2000 tokens length. Also do paragraphing and extract core keywords from the article created.
            prompt = f"The following text is scrapped text from multiple webpages as the results of {self.query}. \n Produce a total professional summary from the text below around 1000 words:\n{pdf_content}\n"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
            )

            # Here, we extract the assistant's message (the last one in the list).
            generated_article = response['choices'][0]['message']['content'].strip()
            
            news_article = f"<h1>Title: The Total Summary of {self.query} </h1> \n <h4> Date: {today} </h4> \n\n <p>{generated_article}<p>"
            
            options = {
                'no-stop-slow-scripts': True,
                'load-error-handling': 'ignore',
                'encoding': "UTF-8",
            }
            path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
            config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
            pdfkit.from_string(news_article, os.path.join(self.folder_path, f"{self.query}_final.pdf"), configuration=config, options=options)


    # def handle_structured_data(self, data):
    #     # Process the scraped data to extract or generate structured data.
    #     # This could include parsing HTML, handling databases or spreadsheets, etc.
    #     pass

    # # Content Generation

    # def personalize_content(self, user_profile, content):
    #     # Adjust the content based on the user's profile or preferences.
    #     # This could involve filtering, sorting, or generating new content.
    #     pass

    # def fact_check(self, content):
    #     # Verify the accuracy of the content. This could involve cross-referencing sources,
    #     # using fact-checking APIs, or similar techniques.
    #     pass

    # def produce_article(self, structured_data):
    #     # Generate a coherent article or report from the structured data.
    #     # This could involve natural language generation techniques.
    #     pass

    # # Testing and Feedback

    # def test_article(self, article):
    #     # Test the generated article for readability, accuracy, relevance, etc.
    #     # This could involve NLP metrics, user feedback, or other tests.
    #     pass

# %%
# TODO insert prompt, show the results by table
# TODO show bs result and prepare pdf to be able to download
# TODO combine with LangChain.DocumentLoader?
# TODO as a side project, get pdf files, give summarized and customized output
# TODO replace to LangChain.Embeddings and LangChain.Vectorstore also FAISS?
# TODO show similarities as right sidebar
# TODO show clustering image as left sidebar
# TODO Show keywords right below the output and then summary followed
# TODO Increase chuck limit with gpt-3.5-turbo-0613
# TODO extract output as pdf download or each companies server for next usage using LangChain.Memory?
# TODO Insert request prompt for article generating process or using OpenAI.FunctionCall?

# %%
# Load the model
acg = AutomatedContentGenerator(api_key='-', cx='-', query='[news] gpt4 usecase in 2023', num_results=3, openai_api_key='-')


# %%
# Step 1: Search and scrape the web
acg.search_and_scrape_web()
acg.save_search_results_to_excel()
acg.scrape_and_save()

# %%
# Step 2 & 3: Process the PDFs and save the embeddings
acg.process_pdfs_and_save_embeddings()
acg.create_clustering_images_from_saved_embeddings()

# %%
# Step 4: Generate the article
acg.process_saved_pdfs()
acg.generate_article()


# %%
# def main():
# # Create an instance of the AutomatedContentGenerator class
#     acg = AutomatedContentGenerator(api_key='AIzaSyCIGuLmSrM65sXknalTE4B4x8PsCZpAZ-I', cx='d4c72f123415549d9', query='chatgpt use cases 2023 news', num_results=5, openai_api_key='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')
#     acg.search_and_scrape_web()
#     acg.save_search_results_to_excel()
#     acg.scrape_and_save()
#     acg.process_pdfs_and_save_embeddings()
#     acg.create_clustering_images_from_saved_embeddings()
#     acg.process_saved_pdfs()
#     acg.generate_article()
    
# if __name__ == '__main__':
#     main()

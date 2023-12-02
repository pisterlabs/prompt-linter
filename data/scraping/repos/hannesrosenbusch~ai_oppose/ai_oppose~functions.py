#cost estimation 

import langchain
from langchain.document_loaders import DataFrameLoader #, PyPDFLoader
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

import io
import os
import shutil
import requests
import time
import re
from datetime import datetime

import numpy as np
import pandas as pd

import openai
import tiktoken
import unidecode

import litstudy
from scholarly import scholarly
from googlesearch import search

from fuzzywuzzy import fuzz
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


def analyze_paragraphs(pdffile: str, openai_model: str, paragraph_length = 1500, max_paragraphs = 5, lower = 0, upper = 10000, left = 0, verbose = False) -> list:
    """
    Function that takes text from pdffile and splits it in sections.
    It loops through sections and checks whether the text is suitable for a scientific review.
    If so it provides a summary and extracts the main claim in the section.
    Theses summaries and claims will be displayed in the review pdf, and they steer the literature search for the critique of this section.
    """
    print("\n---analyzing individual sections in {pdffile}")
    sections = []
    pdftext = trim_pdf(pdffile, lower, upper, left, verbose)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=paragraph_length, chunk_overlap=0, separators=["\n\n", ". "], keep_separator=False)
    docs = text_splitter.create_documents([pdftext])
    if len(docs) == 0:
        print("no readable text found in pdf")
        return []
    limit = np.min([max_paragraphs, len(docs)])
    for doc in docs[0:limit]:
        reviewable_prompt = f"Classify whether the following paragraph is coherent and reviewable. If so, respond with 'yes'. If there are only numbers, references, or unintelligible info, you say 'no'. Paragraph: <<<{doc.page_content}>>>"
        system_prompt = "You are an AI model that determines whether a scientific paragraph carries an argumentation to be reviewed by a scholar. You only say 'yes' or 'no'."
        response = perform_chat_completion(prompt=reviewable_prompt, system_message=system_prompt, temperature=0, openai_model = openai_model)
        if 'no' in response:
            section = {"page_content": doc.page_content, "reviewable": False}
            sections.append(section)
            continue
        #summary
        prompt1 = f"Summarize the following paragraph into a single sentence. Prioritize brevity. If there is no coherent text, just say: ''. Otherwise start your response with 'In this paragraph, the authors...' Paragraph: <<<{doc.page_content}>>>"
        system_prompt1 = "You are an AI model that excels at analyzing research paragraphs."
        response1 = perform_chat_completion(prompt=prompt1, system_message=system_prompt1, temperature=0, openai_model = openai_model)
        #filter summary
        prompt4 = f"Classify whether the following text could be a text summary <<<{response1}>>>. Only answer 'yes' or 'no'"
        system_prompt4 = "You are an AI model that decides whether a text could be a summary of a longer text. You only say 'yes' or 'no'"
        response4 = perform_chat_completion(prompt=prompt4, system_message=system_prompt4, temperature=0, openai_model = openai_model)
        if "no" in response4.lower():
            response1 = "''"
        #main claim
        prompt2 = f"Extract a causal claim from the following paragraph. State the claim within one sentence (Example: 'A low sodium diet reduces the risk of heart disease.'). If there is no clear, scientific claim, just say: ''. Paper snippet: <<<{doc.page_content}>>>"
        system_prompt2 = "You are an AI model that extracts claims from text paragraphs. You state the claim in a concise sentence (Example response: 'Bananas are yellow')."
        response2 = perform_chat_completion(prompt=prompt2, system_message=system_prompt2, temperature=0, openai_model = openai_model)
        #filter claim
        prompt3 = f"Classify whether the following text includes a scientific claim <<<{response2}>>>. Only answer 'yes' or 'no'"
        system_prompt3 = "You are an AI model that decides whether a text is a scientific claim. You only say 'yes' or 'no'"
        response3 = perform_chat_completion(prompt=prompt3, system_message=system_prompt3, temperature=0, openai_model = openai_model)
        if "no" in response3.lower():
            response2 = "''"
        section = {"page_content": doc.page_content, "summary": response1, "claim": response2, "reviewable": True}
        sections.append(section)
    return sections


def autosearch_missing_abstracts(file_path: str, outpath: str) -> pd.DataFrame:
    """"
    this function replaces short (<50) and missing abstracts from the provided csv through calls to google scholar and google.
    abstract source/url is added to output dataframe/file.
    """
    print("\n---auto-web-browsing for missing data\n")
    try:
        df = pd.read_csv(file_path, encoding='utf-8', keep_default_na=False)
    except:
        df = pd.read_csv(file_path, encoding='latin1', keep_default_na=False)
    valid_entries = (df["Author"] != "") & (df["Title"] != "")
    df = df[valid_entries]
    df = df.reset_index(drop=True)
    mask = (df["Abstract Note"].str.len() < 50) | (df["Abstract Note"].isna())
    print(f"""{mask.sum()} abstracts missing out of {len(df)}""")
    use_google = False
    for i in range(len(df)):
        print(f"""\n{shorten(df.loc[i, "Author"], 30)}: {shorten(df.loc[i, "Title"], 40)}""")
        if len(str(df.loc[i, "Abstract Note"])) < 50 or pd.isna(df.loc[i, "Abstract Note"]):
            if not use_google:
                try:
                    new_ref = get_abstract_from_scholar(dict(df.loc[i,:]))
                    if new_ref["Abstract Note"] == "":
                        print("failed to get abstract from scholar-based url. trying google")
                        new_ref = get_abstract_from_google(dict(df.loc[i,:]), bias = ["pubmed", "springer"])
                except Exception as e:
                    use_google = True
            if use_google:
                print("AI OPPOSE MESSAGE: hit google scholar rate limit; using google")
                new_ref = get_abstract_from_google(dict(df.loc[i,:]), bias = ["pubmed", "springer"])
                if new_ref["Abstract Note"] == "":
                    print("try raw google")
                    new_ref = get_abstract_from_google(dict(df.loc[i,:]), bias = [])
            print(new_ref["SOURCE"])
            print(f"""length new abstract: {len(new_ref["Abstract Note"])}""")
            df.loc[i, "Abstract Note"] = new_ref["Abstract Note"]
            df.loc[i, "SOURCE"] = new_ref["SOURCE"]
            if outpath:
                sort_according_to_abstract_len(df).to_csv(outpath, index = False)
    mask = (df["Abstract Note"].str.len() < 50) | (df["Abstract Note"].isna())
    print(f"DONT USE EXCEL FOR ADDING {mask.sum()} MISSING ABSTRACTS! USE FUNCTION >>> manual_entry_abstracts(infile = '{outpath}', outfile = '{outpath}')\n AUTOSEARCH DONE.\n\n")
    return sort_according_to_abstract_len(df)


def create_pdf_from_dicts(sections, output_file):
    print(f"\n---creating {output_file}")
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(name='BoldStyle', parent=styles['Normal'])
    heading_style.fontName = 'Helvetica-Bold'
    heading_style.alignment = 1 # Left alignment
    italic_style = ParagraphStyle(name='ItalicStyle', parent=styles['Normal'])
    italic_style.fontName = 'Helvetica-Oblique'
    italic_style.alignment = 0  # Left alignment
    italic_style_ind = ParagraphStyle(name='ItalicStyle', parent=styles['Normal'], leftIndent=20)
    italic_style_ind.fontName = 'Helvetica-Oblique'
    italic_style_ind.alignment = 0  # Left alignment
    bold_style = ParagraphStyle(name='BoldStyle', parent=styles['Normal'])
    bold_style.fontName = 'Helvetica-Bold'
    bold_style.alignment = 0 # Left alignment
    bold_style_ind = ParagraphStyle(name='BoldStyleInd', parent=styles['Normal'], leftIndent=20)
    bold_style_ind.fontName = 'Helvetica-Bold'
    bold_style_ind.alignment = 0 # Left alignment
    normal_style_ind = ParagraphStyle(name='NormalStyleInd', parent=styles['Normal'], leftIndent=20)
    normal_style_ind.fontName = 'Helvetica'
    normal_style_ind.alignment = 0 # Left alignment
    story = []
    heading = Paragraph("AI OPPOSE: Section-wise Review\n", heading_style)
    story.append(heading)
    story.append(Spacer(1, 20)) 
    intro1 = Paragraph("Dear reader,")
    story.append(intro1)
    story.append(Spacer(1, 5)) 
    intro2 = Paragraph("This AI Bot splits your pdf into sections and reviews them piece-by-piece. Please be aware that the AI does not have the same capabilities as a human reviewer. It might misunderstand or make up things. That is why it always provides the references for you to see.", styles["Normal"])
    story.append(intro2)
    story.append(Spacer(1, 5)) 
    intro3 = Paragraph("Sincerely,")
    story.append(intro3)
    story.append(Spacer(1, 5)) 
    intro4 = Paragraph("AI OPPOSE")
    story.append(intro4)
    story.append(Spacer(1, 4)) 
    current_date = datetime.now().strftime("%B %d, %Y")
    date_location = Paragraph(f"{current_date}<br/>", styles["Normal"])
    story.append(date_location)
    story.append(PageBreak())
    for i, item in enumerate(sections, start=1):
        if not item["reviewable"]:
            continue
        page_content = item.get("page_content", "")
        claim = item.get("claim", "")
        summary = item.get("summary", "")
        opposition = item.get("opposition", "")
        refs = sorted(item.get("references", []))
        section_headline = f"SECTION {i}"
        story.append(Paragraph(section_headline, bold_style))
        story.append(Paragraph("You wrote:", bold_style))
        story.append(Paragraph(f'"{page_content}"', styles["Normal"]))
        story.append(Paragraph("AI summary:", bold_style))
        story.append(Paragraph(f"{summary}", styles["Normal"]))
        story.append(Paragraph("Extracted claim:", bold_style))
        story.append(Paragraph(f"{claim}", styles["Normal"]))
        story.append(Paragraph("AI comments:", bold_style_ind))
        story.append(Paragraph(f"{opposition}", normal_style_ind))
        # responses = sorted(item.get("responses", []))
        # story.append(Spacer(1, 10)) 
        # for r, response in enumerate(responses):
        #     story.append(Paragraph(f"COMMENT {r}: {response}", normal_style_ind))
        # story.append(Spacer(1, 10)) 
        story.append(Paragraph("References:", bold_style_ind))
        for ref in refs:
            story.append(Spacer(1, 4)) 
            story.append(Paragraph(f"- {ref}", italic_style_ind))
        if i < len(sections):
            story.append(PageBreak())
    doc.build(story)


def clean_string(input_string: str, min_word_length = 0, filepath = False, remove_numbers = False, remove_punctuation = True, remove_commas = True) -> str:
    """
    cut punctuation. optionally remove numbers, short words, or everything before last '/' in case of filepath
    """
    if filepath:
        paths = input_string.split("/")
        if len(paths) > 0:
            input_string = paths[-1]
        else:
            input_string = paths[0]
    if remove_punctuation:
        punctuation_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', '-', '–', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
        if remove_commas:
            punctuation_chars.append(",")
        input_string = ''.join(char if char not in punctuation_chars else " " for char in input_string)
    input_string = ' '.join([word for word in input_string.split() if len(word) >= min_word_length])
    if remove_numbers:
        input_string = re.sub(r'\d', '', input_string)
    return input_string


def delete_duplicated_elements(mylist: list) -> list:
    """
    deletes langchain docs (internal usage)
    """
    unique_els = []
    for el in mylist:
        if el not in unique_els:
            unique_els.append(el)
    return unique_els


def delete_duplicated_docs(mylist: list) -> list:
    """
    deletes langchain docs (internal usage)
    """
    unique_els = []
    for el in mylist:
        if el[0] not in [e[0] for e in unique_els]:
            unique_els.append(el)
    return unique_els

    
def extract_abstract(url: str, title: str, authors: str, query_only = False) -> str:
    """
    takes url, paper title, and author name string
    goes to url
    checks for match
    returns abstract or empty string
    """
    title_mismatch = True
    authors_mismatch = True
    chrome_options = Options()
    # chrome_options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=chrome_options)
    try:
        if url.endswith(".pdf") or url.endswith(".PDF"):
            page_text = ""
        elif "annualreviews" in url:
            try:
                driver.get(url)
                driver.implicitly_wait(10)
                all = driver.find_element(By.CLASS_NAME, 'main-content-container')
                page_text = all.find_element(By.CSS_SELECTOR, '[class*="abstractInFull"]').text
                title_found = all.text.split("\n")[1]
                title_mismatch = fuzz.WRatio(clean_string(title), title_found) < 85
                authors_found = clean_string(driver.find_element(By.CLASS_NAME, 'name').text.replace(" and ", " "), remove_numbers=True, remove_commas=False).replace(",,", ",")
                authors_found_clean = clean_string(authors_found, remove_numbers=True, min_word_length=2)
                authors_mismatch = fuzz.WRatio(authors, authors_found_clean) < 70
                driver.quit()
            except Exception as e:
                page_text = ""
        elif "onlinelibrary.wiley" in url:
            try:
                chrome_options = Options()
                driver = webdriver.Chrome(options=chrome_options)
                driver.get(url)
                driver.implicitly_wait(10)
                page_div = driver.find_element(By.CLASS_NAME, 'article__body ')
                page_text = page_div.find_element(By.CSS_SELECTOR, "section[lang='en']").text
                title_found = clean_string(driver.find_element(By.CLASS_NAME, 'citation__title').text)
                title_mismatch = fuzz.WRatio(clean_string(title), title_found) < 85
                authors_found = driver.find_element(By.ID, 'sb-1').text.replace("\n","")
                authors_found_clean = clean_string(authors_found, remove_numbers=True, min_word_length=2)
                authors_mismatch = fuzz.WRatio(authors, authors_found_clean) < 70
                driver.quit()
            except Exception as e:
                page_text = ""
        elif "journals.plos" in url:
            try:
                driver.get(url)
                driver.implicitly_wait(10)
                title_found = clean_string(driver.find_element(By.ID, 'artTitle').text)
                page_text = driver.find_element(By.CLASS_NAME, 'abstract-content').text
                title_mismatch = fuzz.WRatio(clean_string(title), title_found) < 85
                authors_found = driver.find_element(By.ID, 'author-list').text.replace("\n","")
                authors_found_clean = clean_string(authors_found, remove_numbers=True, min_word_length=2)
                authors_mismatch = fuzz.WRatio(authors, authors_found_clean) < 70
                driver.quit()
            except Exception as e:
                page_text = ""
        elif "royalsocietypublishing" in url:
            try:
                driver.get(url)
                driver.implicitly_wait(10)
                title_found = clean_string(driver.find_element(By.CLASS_NAME, 'citation__title').text)
                page_div = driver.find_element(By.CLASS_NAME, 'article__body ')
                page_text = page_div.find_element(By.CLASS_NAME, "hlFld-Abstract").text 
                title_mismatch = fuzz.WRatio(clean_string(title), title_found) < 85
                authors_found = driver.find_element(By.ID, 'sb-1').text.replace("\n","")
                authors_found_clean = clean_string(authors_found, remove_numbers=True, min_word_length=2)
                authors_mismatch = fuzz.WRatio(authors, authors_found_clean) < 70
                driver.quit()
            except Exception as e:
                page_text = ""
        elif "biomedcentral" in url:
            try:
                driver.get(url)
                driver.implicitly_wait(10)
                title_found = clean_string(driver.find_element(By.CLASS_NAME, 'c-article-title').text)
                page_text = driver.find_element(By.CSS_SELECTOR, "section[data-title='Abstract']").text
                title_mismatch = fuzz.WRatio(clean_string(title), title_found) < 85
                author_divs = driver.find_elements(By.CLASS_NAME,"c-article-author-list__item")
                authors_found = ", ".join([clean_string(a.text, min_word_length=2) for a in author_divs])
                authors_mismatch = fuzz.WRatio(authors, clean_string(authors_found, min_word_length=2)) < 70
                driver.quit()
            except Exception as e:
                page_text = ""
        elif "pubmed.ncbi" in url:
            try:
                driver.implicitly_wait(10)
                page_text = driver.find_element(By.ID, 'eng-abstract').text
                title_found = clean_string(driver.find_element(By.CLASS_NAME, 'heading-title').text)
                title_mismatch = fuzz.WRatio(clean_string(title), title_found) < 85
                author_divs = driver.find_elements(By.CLASS_NAME,"authors-list-item")
                authors_found = ", ".join([clean_string(a.text, remove_numbers=True, min_word_length=2) for a in author_divs])
                authors_mismatch = fuzz.WRatio(authors, clean_string(authors_found, min_word_length=2)) < 70
                driver.quit()
            except Exception as e:
                page_text = ""
        elif "link.springer" in url or "www.nature" in url:
            try:
                driver.get(url)
                driver.implicitly_wait(10)
                parent_div = driver.find_element(By.CLASS_NAME, 'c-article-body')
                page_text = parent_div.find_element(By.CSS_SELECTOR, "section[data-title='Abstract']").text
                title_found = clean_string(driver.find_element(By.CLASS_NAME, 'c-article-title').text)
                title_mismatch = fuzz.WRatio(clean_string(title), title_found) < 85
                author_divs = driver.find_elements(By.CLASS_NAME,"c-article-author-list__item")
                authors_found = ", ".join([clean_string(a.text, min_word_length=2) for a in author_divs])
                authors_mismatch = fuzz.WRatio(authors, clean_string(authors_found, min_word_length=2)) < 70
                driver.quit()
            except Exception as e:
                page_text = ""
        elif "researchgate" in url:
            try:
                driver.get(url)
                driver.implicitly_wait(10)
                abstract_divs = driver.find_elements(By.CSS_SELECTOR, '[class*="abstract"]')
                page_text = [abstract_div.text for abstract_div in abstract_divs if len(abstract_div.text) > 50][0]
                title_found = clean_string(driver.find_element(By.CSS_SELECTOR, '[class*="__title"]').text)
                title_mismatch = fuzz.WRatio(clean_string(title), title_found) < 85
                author_divs = driver.find_elements(By.CLASS_NAME,"publication-header-author")
                authors_found = [d.find_element(By.CSS_SELECTOR, '[class*="__title"]').text for d in author_divs]
                authors_found = ", ".join([clean_string(a, min_word_length=2) for a in authors_found])
                authors_mismatch = fuzz.WRatio(authors, clean_string(authors_found, min_word_length=2)) < 70
                driver.quit()
            except Exception as e:
                page_text = ""
        elif "frontiersin.org" in url:
            try:
                driver.get(url)
                driver.implicitly_wait(10)
                abstr_div = driver.find_element(By.CLASS_NAME, 'JournalAbstract')
                page_text = abstr_div.find_element(By.TAG_NAME, 'p').text
                title_found = clean_string(driver.find_element(By.TAG_NAME, 'h1').text)
                title_mismatch = fuzz.WRatio(clean_string(title), title_found) < 85
                authors_found = abstr_div.find_element(By.CLASS_NAME,"authors").text.replace("*", "")
                authors_mismatch = fuzz.WRatio(authors, clean_string(authors_found, min_word_length=2)) < 70
                driver.quit()
            except Exception as e:
                page_text = ""
        elif "sciencedirect" in url:
            try:
                driver.get(url)
                driver.implicitly_wait(10)
                abstract_divs = driver.find_elements(By.CSS_SELECTOR, '[class*="abstract"]')
                page_text = [abstract_div.text for abstract_div in abstract_divs if len(abstract_div.text) > 60][0]
                title_found = clean_string(driver.find_element(By.CLASS_NAME, 'title-text').text)
                title_mismatch = fuzz.WRatio(clean_string(title), title_found) < 85
                authors_found = driver.find_element(By.CLASS_NAME,"author-group").text.replace("Author links open overlay panel", "")
                authors_mismatch = fuzz.WRatio(authors, clean_string(authors_found, min_word_length=2)) < 70
                driver.quit()
            except Exception as e:
                page_text = ""
        elif "psycnet" in url:
            try:
                driver.get(url)
                driver.implicitly_wait(10)
                page_text =  driver.find_element(By.TAG_NAME, 'abstract').text
                title_found = clean_string(driver.find_element(By.CLASS_NAME, 'm-t-0').text)
                title_mismatch = fuzz.WRatio(clean_string(title), title_found) < 85
                authorsdivs = driver.find_elements(By.CLASS_NAME,"linked-author")
                authors_found = ", ".join([a.text for a in authorsdivs])
                authors_mismatch = fuzz.WRatio(authors, clean_string(authors_found, min_word_length=2)) < 70
                driver.quit()
            except Exception as e:
                print(e)
                page_text = ""
        elif "www.ncbi." in url:
            try:
                driver.get(url)
                driver.implicitly_wait(10)
                try:
                    page_text = driver.find_element(By.XPATH, "//*[contains(text(), 'Abstract')]/following-sibling::*[1]").text
                except:
                    page_text = driver.find_element(By.XPATH, "//*[contains(text(), 'Summary')]/following-sibling::*[1]").text
                title_found = clean_string(driver.find_element(By.CLASS_NAME, 'content-title').text)
                title_mismatch = fuzz.WRatio(clean_string(title), title_found) < 85
                preceding_element = driver.find_element(By.CLASS_NAME, 'content-title')
                following_div = preceding_element.find_element(By.XPATH, "following-sibling::div")
                authors_found = following_div.text
                authors_mismatch = fuzz.WRatio(authors, clean_string(authors_found, min_word_length=2)) < 70
                driver.quit()
            except Exception as e:
                print(e)
                page_text = ""
        elif "journals.sagepub" in url:
            try:
                driver.get(url)
                driver.implicitly_wait(10)
                try:
                    page_text = driver.find_element(By.XPATH, "//*[contains(text(), 'Abstract')]/following-sibling::*[1]").text
                except:
                    page_text = driver.find_element(By.XPATH, "//*[contains(text(), 'Summary')]/following-sibling::*[1]").text
                title_found = clean_string(driver.find_element(By.TAG_NAME, 'h1').text)
                title_mismatch = fuzz.WRatio(clean_string(title), title_found) < 85
                authors_found = driver.find_element(By.CLASS_NAME, 'authors').text.replace(" and ", " ")
                authors_found = ", ".join([a for a in authors_found.split(" ") if "@" not in a])
                authors_mismatch = fuzz.WRatio(authors, clean_string(authors_found, min_word_length=2)) < 70
                driver.quit()
            except Exception as e:
                print(e)
                page_text = ""
        elif "aeaweb.org" in url:
            try:
                driver.get(url)
                driver.implicitly_wait(10)
                title_found = clean_string(driver.find_element(By.CLASS_NAME, 'title').text)
                title_mismatch = fuzz.WRatio(clean_string(title), title_found) < 85
                authors_found = driver.find_element(By.CLASS_NAME, 'attribution').text
                authors_mismatch = fuzz.WRatio(authors, clean_string(authors_found, min_word_length=2)) < 70
                article = driver.find_element(By.ID, "article-information").text
                page_text = article.split("Citation")[0]
                driver.quit()
            except Exception as e:
                print(e)
                page_text = ""
        else:
            page_text = ""
        if page_text == "":
            print(url)
        if not query_only and (title_mismatch or authors_mismatch):
            page_text = ""
        if page_text.lower().startswith("abstract\n"):
            page_text = page_text[len("abstract\n"):]
    except Exception as e:
        print(e)
        page_text = ""
    if query_only:
        if page_text != "":
            return {"Author": authors_found, "Title": title_found, "Publication Year": None, "Abstract Note": page_text.replace("\n", " "), "SOURCE": url}
        else:
            return {"Author": "", "Title": "", "Publication Year": None, "Abstract Note": "", "SOURCE": url}
    else:
        return page_text.replace("\n", " ")


def extract_focal_claims(pdf_path: str, openai_model: str, max_claims = 3, lower = 0, upper = 10000, left = 0, verbose = False) -> list:
    """"
    DEPRECATED
    takes a pdf and returns {max_claims} novel claims made in the paper.
    """
    pdftext = trim_pdf(pdf_path, lower, upper, left, verbose)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3800, chunk_overlap=200)
    docs = text_splitter.create_documents([pdftext])
    claims = []
    nr_claims = 0
    for i, doc in enumerate(docs):
        if nr_claims >= max_claims:
            break
        prompt = f"Extract the novel claim from the following research paper snippet. Only give the claim that is novel and unique to the publication. Ignore claims from previous references mentioned in the paper. Simply reproduce the novel claim within one sentence (Example responses: 'A low sodium diet reduces the risk of heart disease.', 'The attention mechanism was not pioneered in transformer architectures.'). If there is no novel claim, just say: ''. Paper snippet: <<<{doc.page_content}>>>"
        system_prompt = "You are an AI model that extracts the central claim from research papers. You state the claim in a single sentence as if you were the original author."
        response = perform_chat_completion(prompt= prompt, system_message=system_prompt, temperature=0, openai_model = openai_model)
        if response != "" and response != "''":
            claims.append(response)
            nr_claims += 1
    return claims


def extract_last_names(text: str) -> str:
    """
    DEPRECATED
    extracts last names by assuming they are followed by a comma"""
    if "," in text:
        words = text.split(" ")
        text = " ".join(name for name in words if name[-1] == ",")
    return clean_string(text, min_word_length=2)

    

def extract_pdf_references(filepath: str, lower: int, upper: int, left: int, verbose: bool, openai_model: str) -> list:
    """
    takes pdf filepath to a research paper and uses gpt to extract list of references
    """
    pdftext = trim_pdf(filepath, lower, upper, left, verbose)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=200)
    docs = text_splitter.create_documents([pdftext])
    if len(docs) == 0:
        print("no readable text found in pdf")
        return []
    responses = []
    system_prompt = "You are an AI tool that extracts reference lists from papers."
    no_hits = 0
    found_refs = False
    for doc in reversed(docs):
        prompt = f"Extract author names, paper titles, and years from the following bibliography. Respond in this list format: [{{'year': '1992', 'authors': 'E. Wiltens, A. Stone', 'title': 'effects of mdma'}}, {{'year': '2022', 'authors': 'S. Wiesen', 'title': 'durability of iron in water'}}]. Exclude journal names. Return an empty list [] if there was no bibliography or only short in-text references without paper titles. TEXT: {doc.page_content}"
        response = perform_chat_completion(prompt= prompt, system_message=system_prompt, temperature=0, openai_model = openai_model)
        ref_list = references_string_as_list(response)
        if len(ref_list) > 0:
            found_refs = True
        else:
            if found_refs: #if there were refs found previously then we might have gone up to the beginning of the bibliography
                no_hits += 1
                if no_hits == 2:
                    break
            continue
        responses.extend(ref_list)
    try:
        responses = [ref for ref in responses if ref["title"] != "" and ref["authors"] != ""]
        responses.sort(key= lambda x: clean_string(x["authors"], min_word_length=2))
        new_key_names = {"title": "Title", "year": "Publication Year", "authors": "Author"}
        responses = [rename_dict_keys(ref, new_key_names) for ref in responses]
    except:
        print("likely the LLM misformatted the pdf references")
    return delete_duplicated_elements(responses)


def find_metadata_for_ref(ref: dict) -> dict:
    """
    takes in a ref dictionary with Title, Publication Year, and Author
    searches for that ref with litstudy
    if match found, it adds Abstract, References, and Citations to the ref dictionary
    """
    docs = litsearch(ref["Title"])
    if ref["Title"] != "" and ref["Title"] is not None:
        for doc in docs:
            if doc.authors in [None,'N/A'] or doc.publication_year in [None,'N/A'] or doc.title in [None,'N/A']:
                continue
            title_orig = clean_string(ref["Title"].lower(), min_word_length=2)
            title_search = clean_string(doc.title.lower(), min_word_length=2)
            titles_match = fuzz.WRatio(title_orig, title_search) > 85
            authors_orig = clean_string(ref["Author"].lower(), min_word_length=2)
            authors_search = clean_string(' '.join([doc.authors[i].name.lower() for i in range(len(doc.authors))]), min_word_length=2)
            authors_match = fuzz.WRatio(authors_orig, authors_search) > 70
            if is_integer(ref["Publication Year"]):
                year_match = np.abs(int(doc.publication_year) - int(ref["Publication Year"])) <= 1
            else: 
                year_match = False
            if year_match and authors_match and titles_match:
                ref = {"Author": ', '.join([doc.authors[i].name.lower() for i in range(len(doc.authors))]), "Title": doc.title, "Publication Year": doc.publication_year, "Abstract Note": doc.abstract, "References": [], "Citations": []}
                if doc.references:
                    ref["References"] =  [d for d in doc.references]
                if doc.citations:
                    ref["Citations"] = [d for d in doc.citations]
                return ref
    #no match
    ref["Abstract Note"] = ""
    ref["References"] = []
    ref["Citations"] = []
    return ref


def filter_docs(presented_claim: str, relevant_docs: list, openai_model: str) -> list:
    """
    This function checks whether the docs returned from the literature search are truly relevant. (internal usage)
    """
    openai.api_key = os.environ['OPENAI_API_KEY']
    filtered_docs = []
    for doc, score in relevant_docs:
        filterprompt = f"""
        Presented snippet:
        {presented_claim}

        Target paper:
        {doc.page_content}

        Instructions:
        Classify whether the Target Paper is relevant to the Presented Snippet. Answer "yes" or "no". If unsure, answer "no".
        """
        filterresponse = perform_chat_completion(filterprompt, system_message="You are an attentive academic machine that only says 'yes' or 'no'.", temperature=0, openai_model = openai_model)
        if "yes" in filterresponse.lower():
            filtered_docs.append((doc, score))
        time.sleep(1)
    return filtered_docs


def format_texts_from_csv(file_path: str, abstract_column: str, author_column: str, title_column: str, year_column: str) -> pd.DataFrame:
    """
    columns expected in csv (with default zotero names):
    "Abstract Note": abstract of the paper
    "Publication Year": year of the publication
    "Author": assumes last names are followed by comma
    "Title": title of paper
    """
    try:
        data = pd.read_csv(file_path, encoding='utf-8', keep_default_na=False)
    except:
        data = pd.read_csv(file_path, encoding='latin1', keep_default_na=False)
    chosen_cols = [abstract_column, author_column, title_column, year_column]
    if not all([col in data.columns for col in chosen_cols]):
        raise ValueError(f"make sure that your csv has the chosen column headers: {chosen_cols}")
    data = data[data[abstract_column] != '']
    indices = data[data[abstract_column].isna() | data[abstract_column].eq('')].index
    if len(indices > 0):
        print(f'careful there are empty cells in column "{abstract_column}" resulting in empty literature entries.\nROWS: {indices}')
    for col in chosen_cols:
        data[col] = data[col].astype(str)
    data[author_column] = data[author_column].apply(lambda x: x.replace(".", ""))
    data[year_column] = data[year_column].replace('', 'n.d.')
    data[author_column] = data[author_column].apply(lambda x: clean_string(x, min_word_length=2, remove_punctuation=False))
    data[author_column] = data[author_column].str.title()

    data["file_content"] = "TITLE: '" + data[title_column] + "' ABSTRACT: '" + data[abstract_column] + "' SOURCE: " + "("  + data[author_column]  + ", " + data[year_column].astype(str) + ")"
    data["reference"] = [shorten(author, 120) + " ("  + str(yea) + "). " for author, yea in zip(data[author_column], data[year_column].astype(str) )]
    if "SOURCE" in data.columns:
        data["reference"] = data["reference"] + data["Title"] + '. | Retrieved from ' + data["SOURCE"]
    data["reference"] = data["reference"].apply(unidecode.unidecode)
    data["file_content"] = data["file_content"].apply(unidecode.unidecode)
    return data


def generate_adversarial_texts(presented_text: str, relevant_docs: list, openai_model: str,  summarize_results = True, input_type = "claim") -> tuple:
    """
    This function produces the adversarial claims through openai and under reference of the relevant documents.
    input_type: 'claim' or 'text'
    """
    openai.api_key = os.environ['OPENAI_API_KEY']
    responses = []
    for doc in relevant_docs:
        prompt1 = f"""
        Presented {input_type}:
        {presented_text}

        Previous paper:
        {doc.page_content}

        Instructions:
        Refine, criticize, or oppose the presented {input_type}. If you refer to the previous paper, provide the apa reference. Use two concise sentences.
        """
        response = perform_chat_completion(prompt1, f"You are a reviewer of an academic {input_type}; you stick to the facts from the literature.", temperature=0, openai_model = openai_model)
        responses.append({"response": response, "metadata": doc.metadata["reference"], "page_content": doc.page_content})
    if summarize_results and len(responses) > 0:
        nr_tokens = 0
        all_comments = ""
        for response in responses:
            all_comments = f"""{all_comments}\nComment: {response["response"]}"""
        summary_prompt = f"""
        Presented {input_type}:
        {presented_text}

        Your comments:
        {all_comments}

        Instructions:
        Summarize your comments to the {input_type} into a coherent text. Keep all the relevant apa references as in-text references (no full bibliography). Keep it concise.
        """
        summary_response = perform_chat_completion(summary_prompt, f"You are a reviewer of an academic {input_type}; you stick to the facts from the literature.", temperature=0, openai_model = openai_model)
    else:
        summary_response = None
    return (responses, summary_response)


def generate_dict_from_pdf(pdf_path: str, openai_model: str, max_len = 3500, split_on_abstract = True, only_abstract = True) -> any:
    """
    used to extract research abstract from (URL to) pdf file
    """
    pdf_text = ''
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    if len(pdf_reader.pages) == 0:
        raise ValueError(f"no readable pdf: {pdf_path}")
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()
    abstract = pdf_text.lower().replace("\n","")
    if split_on_abstract:
        pdf_chunks = abstract.split("abstract")
        if len(pdf_chunks) > 1:
            abstract = pdf_chunks[1]
    if len(abstract) >= max_len:
        abstract = abstract[:max_len]
    if only_abstract:
        return abstract
    sysmes = "You are an AI that simply extracts the desired information from texts. No preamble like 'The answer is 2021' just '2021'"
    title = perform_chat_completion(f"Please extract the title of this paper from the snippet. If there is no title just say 'no title'. SNIPPET >>> {pdf_text[0:4000]}", sysmes, openai_model = openai_model)
    authors = perform_chat_completion(f"Please extract the names of the authors of this paper from the snippet. If there are no authors mentioned, just say 'no authors'. SNIPPET >>> {pdf_text[0:4000]}", sysmes, openai_model = openai_model)
    year = perform_chat_completion(f"Please extract the publication year from the article snippet. If there is no year mentioned, just say 'n.d.'. SNIPPET >>> {pdf_text[0:4000]}", sysmes, openai_model = openai_model)
    return {"Author": authors, "Title": title, "Publication Year": year, "Abstract Note": abstract, "SOURCE": pdf_path}
    



def generate_literature_csv_from_pdf(pdf_path: str, openai_model: str, author_column: str, title_column: str, year_column: str, max_secondary: int, output_path = None,  lower = 0, upper = 10000, left = 0, verbose = False, include_source_as_ref = False) -> pd.DataFrame:
    """
    large wrapper around ref extraction from pdf and ref extension with secondary literature
    """
    print(f"\n---extracting references from {pdf_path}\n")
    refs = extract_pdf_references(pdf_path, lower, upper, left, verbose, openai_model)
    if len(refs) == 0:
        return None
    data = pd.DataFrame(refs)
    data = remove_duplicate_rows(data)
    print(f"  --> {len(data)} refs extracted")
    print("\n---searching for additional papers based on Abstract\n")
    abstract = generate_dict_from_pdf(pdf_path, openai_model, only_abstract=True)
    all_finds = []
    forbidden_terms = []
    for i in range(2):
        query = perform_chat_completion(prompt=f"come up with a relevant search query (max 6 words) for the provided abstract (e.g., 'sleep AND depression'). the goal is to search for papers critical of the abstract! if unsure just say: ''. The abstract <<<{abstract}>>>", system_message=f"You are an AI model that generates searchterms. Forbidden terms: {forbidden_terms}", openai_model = openai_model)
        forbidden_terms.append(query)
        finds = get_abstract_from_google({"query": f"paper {query}"}, bias = [], query_only = True)
        all_finds.extend(finds)
    if include_source_as_ref:
        primary_ref = generate_dict_from_pdf(pdf_path, openai_model, only_abstract=False)
        all_finds.append(primary_ref)
    data_google = pd.DataFrame(all_finds)
    data = pd.concat([data, data_google], ignore_index=True)
    valid_entries = (data["Author"] != "") & (data["Title"] != "")
    data = data[valid_entries]
    data = data.reset_index(drop=True)
    data = remove_duplicate_rows(data)
    print("\n---finding abstracts\n")
    print("\n---finding related papers\n")
    finds = []
    entry_counts = {}
    for i in range(data.shape[0]):
        ref = {"Author": data[author_column][i], "Title": data[title_column][i], "Publication Year": data[year_column][i]}
        find = find_metadata_for_ref(ref)
        finds.append(find)
        secondary_references = [r.doi for r in find["References"]]
        secondary_citations = [r.doi for r in find["Citations"]]
        ref_candidates = secondary_references + secondary_citations
        for entry in ref_candidates:
            entry_counts[entry] = entry_counts.get(entry, 0) + 1
    data_enriched = pd.DataFrame(finds)
    data_enriched["SOURCE"] = f"{pdf_path}"
    selected_columns = [author_column, title_column, year_column, "Abstract Note"]
    filtered_secondary = {key: value for key, value in entry_counts.items() if key is not None}
    max_secondary = np.min([max_secondary, len(filtered_secondary.keys())])
    filtered_secondary = dict(sorted(filtered_secondary.items(), key=lambda item: item[1], reverse=True)[:max_secondary])
    second_order_finds = []
    print("\n---finding abstracts of related papers\n")
    for doi in filtered_secondary:
        try:
            second_order_find = {}
            t = litstudy.fetch_semanticscholar(doi)
            second_order_find["Title"] = t.title
            second_order_find["Author"] = ', '.join([t.authors[i].name for i in range(len(t.authors))])
            second_order_find["Publication Year"] = t.publication_year
            second_order_find["Abstract Note"] = t.abstract
            second_order_find["SOURCE"] = doi
            second_order_finds.append(second_order_find)
        except:
            print(f"could not retrieve doi: {doi}")
    data_secondary_refs = pd.DataFrame(second_order_finds)
    combined_df = pd.concat([data_enriched[selected_columns + ["SOURCE"]], data_secondary_refs], ignore_index=True)
    combined_df['Abstract Note'] = combined_df['Abstract Note'].fillna('')
    combined_df['Abstract Note Length'] = combined_df['Abstract Note'].str.len()
    combined_df = combined_df.sort_values(by='Abstract Note Length', ascending=True)
    combined_df = combined_df.drop(columns=['Abstract Note Length'])
    combined_df = combined_df.reset_index(drop=True)
    combined_df_clean = remove_duplicate_rows(combined_df)
    if output_path:
        combined_df_clean.to_csv(output_path, index = False)
    return combined_df_clean


def generate_vectorstore(data: pd.DataFrame, filepath: str, max_doc_size = 4000) -> any:
    """generates a folder that should not be deleted as it will be used for the claim oppositions later. 
    it costs money to use this function as it is run with openai embedding model"""
    print("\n---generating AI vectorstore.\n")
    ### search for vectorstore in directory
    # booleans = ["vectorstore_" in f for f in os.listdir()]
    # if any(booleans):
    #     existing_vectorstore = [f for f, b in zip(os.listdir(),booleans) if b][0]
    #     userinput = input(f"Would you like to work with {existing_vectorstore} instead of generating a new vectorstore? (y/n)")
    #     if "y" in userinput.lower():
    #         return existing_vectorstore
    filepath = shorten(clean_string(filepath,  filepath=True).replace("csv", ""), 40)
    new_chroma_directory = f'vectorstore_{filepath}_{time.ctime().replace(":", "").replace(" ", "")[0:-4]}'
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    loader = DataFrameLoader(data[["file_content","reference"]], page_content_column = "file_content")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_doc_size, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    if len(data) != len(data):
            print(f"Careful! {len(data)} files were saved as {len(docs)} documents in the vector store.")
    if os.path.exists(new_chroma_directory) and os.path.isdir(new_chroma_directory):
        if len(os.listdir(new_chroma_directory)) > 0:
            print("Careful! You already seem to have files in your new_chroma_directory. You might create duplicates which affect search results.")
    docsearch = Chroma.from_documents(docs, embeddings, persist_directory=new_chroma_directory)
    docsearch.persist()
    print(f"vectorstore has been generated: folder {new_chroma_directory}\n\n")
    return new_chroma_directory


def get_abstract_from_google(ref: dict, bias: list, query_only = False)-> any:
    """
    takes ref dictionary, googles ref to get paper URL, gets abstract from url
    """
    if query_only:
        ref_dicts = []
        urls = search_with_backoff(f"""{ref["query"]}""", num_results = 10)
        for url in urls:
            dic = extract_abstract(url, "", "", query_only=True)
            ref_dicts.append(dic)
        return ref_dicts
    else:
        ref["Abstract Note"] = ""
        if f"""{clean_string(ref["Author"], min_word_length = 2)}""" == "":
            return ref
        # try:
        short_author = f"""{clean_string(ref["Author"], min_word_length = 2)}""".split()[0]
        url = search_with_backoff(f"""{' '.join(bias)} {short_author} {ref["Title"]}""", num_results=1)
        time.sleep(np.random.randint(15,20))
        if url:
            ref["Abstract Note"] = extract_abstract(url, title = clean_string(ref["Title"]), authors = clean_string(ref["Author"], min_word_length=2))
            if len(ref["Abstract Note"]) > 20:
                ref["SOURCE"]  =  ref["SOURCE"].split("-->")[0] + f' --> {url}'
        return ref


def get_abstract_from_scholar(ref: dict) -> dict:
    """
    takes ref dictionary, searches scholar to get paper URL, gets abstract from url
    """
    ref["Abstract Note"] = ""
    results = scholarly.search_single_pub(ref["Title"]) #search_pubs
    time.sleep(np.random.randint(15,20))
    try:
        first_result = results #next(results)
    except StopIteration as e:
        try:
            short_title = shorten(f"""{clean_string(ref["Title"], min_word_length = 2)}""", 25)
            short_author = f"""{clean_string(ref["Author"], min_word_length = 2)}""".split()[0]
            results = scholarly.search_single_pub(f"{short_author} {short_title}")
            first_result = results
        except:
            ref["SOURCE"] += f"--> NO RESULTS ON SCHOLAR"
            return ref
    try:
        url = first_result["pub_url"]
        title = first_result["bib"]["title"]
        year = first_result["bib"]["pub_year"]
        authors = first_result["bib"]["author"]
    except:
        ref["SOURCE"] += f"--> NO VALID URL ON SCHOLAR"
        return ref
    title_orig = clean_string(ref["Title"].lower(), min_word_length=2)
    title_search = clean_string(title.lower(), min_word_length=2)
    titles_match = fuzz.WRatio(title_orig, title_search) > 85
    authors_orig = clean_string(ref["Author"].lower(), min_word_length=2)
    authors_search = clean_string(' '.join([authors[i].lower() for i in range(len(authors))]), min_word_length=2)
    authors_match = fuzz.WRatio(authors_orig, authors_search) > 70
    if is_integer(ref["Publication Year"]) and is_integer(year):
        year_match = np.abs(int(year) - int(ref["Publication Year"])) <= 1
    else: 
        year_match = True
    print(f"scholar found paper: {all([titles_match, authors_match, year_match])}")
    if titles_match and authors_match and year_match:
        ref["Abstract Note"] = extract_abstract(url, title = clean_string(ref["Title"]), authors = clean_string(ref["Author"]))
        if len(ref["Abstract Note"]) > 20:
            ref["SOURCE"] = ref["SOURCE"].split("-->")[0] + f' --> {url}'
    else:
        ref["SOURCE"] += f"--> SCHOLAR RESULT LOOKS LIKE MISMATCH"
    return ref


def get_relevant_docs(section: dict, openai_model: str, chroma_directory: str, max_documents = 8, search_width = 2) -> any:
    """
    conducts various similarity searches between the given claim and the literature in the vectorstore.
    returns relevant langchain documents.
    """
    if not section["reviewable"]:
        print("not reviewable, so no document search")
        return []

    openai.api_key = os.environ['OPENAI_API_KEY']
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    vectordb = Chroma(persist_directory=chroma_directory, embedding_function=embeddings)
    relevant_docs = []

    # #V1 full section
    queryV1 = f"""The following text is incorrect: "{section["page_content"]}" Other papers present opposing arguments."""
    relevant_docs.extend(vectordb.similarity_search_with_score(queryV1, k=search_width))

    #V2 full section paraphrazed
    relevant_docs = delete_duplicated_docs(relevant_docs)
    queryV2 = perform_chat_completion(f'This is an academic text snippet: "{section["page_content"]}". Paraphraze the text. Keep the meaning roughly the same but exchange lots of words.', "You are a good writer that is able to edit and revise text as instructed.", temperature=0, openai_model = openai_model)
    relevant_docs.extend(vectordb.similarity_search_with_score(queryV2, k=len(relevant_docs) + 1))
    relevant_docs = delete_duplicated_docs(relevant_docs)

    if section["claim"] != "''":
        #V3 claim reversal
        promptV3 = f"""Someone wrote this claim: "{section["claim"]}" Write an opposing statement in which you use different scientific words but no sources."""
        queryV3 = perform_chat_completion(promptV3, 'You are a talented academic AI that confidently opposes scientific claims.', openai_model = openai_model)
        relevant_docs.extend(vectordb.similarity_search_with_score(queryV3, k=search_width))
        #V4 claim meta-analyses and reviews
        queryV4 = f"""Meta-analyses and reviews on this: '{section["claim"]}'"""
        relevant_docs.extend(vectordb.similarity_search_with_score(queryV4, k=len(relevant_docs) + 1))
    relevant_docs = delete_duplicated_docs(relevant_docs)

    if section["summary"] != "''":
        #V5 summary
        relevant_docs = delete_duplicated_docs(relevant_docs)
        queryV5 = f"""Doubt surrounds this idea: {section["summary"]}"""
        relevant_docs.extend(vectordb.similarity_search_with_score(queryV5, k=len(relevant_docs) + 1))
        #V6 summary paraphrazation
        relevant_docs = delete_duplicated_docs(relevant_docs)
        responseV6 = perform_chat_completion(f'Write a concise paraphrazation of the following text. Change as many words as possible: "{section["summary"]}".', "You are a good writer that is able to edit and revise text as instructed.", temperature=0, openai_model = openai_model)
        queryV6 = f"These ideas have been reviewed and criticized: '{responseV6}'"
        relevant_docs.extend(vectordb.similarity_search_with_score(queryV6, k=len(relevant_docs) + 1))

    relevant_docs = delete_duplicated_docs(relevant_docs)
    relevant_docs = filter_docs(section["page_content"], relevant_docs, openai_model)

    if max_documents is not None:
        if not isinstance(max_documents, int):
            raise ValueError(f"max_documents must be int, but is {max_documents}")
        relevant_docs = sorted(relevant_docs, key=lambda x: x[1]) #prioritize high similarity results
        if max_documents < len(relevant_docs):
            relevant_docs = relevant_docs[0:max_documents]
    relevant_docs = [doc[0] for doc in relevant_docs]
    if len(relevant_docs) == 0:
        print(f"\nNo relevant documents found in your literature corpus '{chroma_directory}'! Try adding papers.\n")
        for doc in relevant_docs:
            doc.page_content = "No documents were relevant for the claim. No comment can be generated."
            doc.metadata = ""
    return relevant_docs


def get_example_literature_file(lit = "egodepletion"):
    """
    possible values for lit argument are egodepletion, humorXdating, powerpose
    """
    if lit.lower() == "egodepletion":
        filename = "egodepletion_literature.csv"
    elif lit.lower() == "humorxdating":
        filename = "humorXdating_literature.csv"
    elif lit.lower() == "powerpose":
        filename = "powerpose_literature.csv"
    else:
        raise ValueError("lit argument should be one of 'egodepletion', 'humorXdating', 'powerpose'")
    data_directory = os.path.join(os.path.dirname(__file__), 'data')
    source_path = os.path.join(data_directory, filename)
    destination_path = filename
    print(f"collected file -> {destination_path}")
    shutil.copy(source_path, destination_path)


def insert_linebreaks(text: str, interval: int) -> str:
    """
    string manipulation helper (internal usage)
    """
    lines = []
    current_line = ""
    for word in text.split():
        if word in ["ABSTRACT:", "SOURCE:"]:
            lines.append(current_line.strip())
            current_line = word + " "
        elif len(current_line) + len(word) <= interval:
            current_line += word + " "
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())
    return '\n'.join(lines)


def is_integer(val: any) -> bool:
    """
    checks whether string can be converted to valid integer
    """
    try:
        int(val)
        return True
    except:
        return False


def litsearch(q: str, max_results = 3) -> list:
    """
    accepts query string
    uses litstudy package to query semanticscholar/crossfref
    returns list of results
    """
    q = clean_string(q)
    docs = []
    if q != "" and q is not None:
        try:
            docs = litstudy.search_semanticscholar(q, limit = max_results, batch_size=max_results)
        except:
            try:
                q_words = q.split(" ")
                q_shortened = " ".join(q_words[0:int(len(q_words)/2)])
                docs = litstudy.search_semanticscholar(q_shortened, limit = max_results, batch_size=max_results)
            except:
                try:
                    docs = litstudy.search_crossref(q, limit = max_results)
                except:
                    print(f"search fail for semantic scholar and crossref: '{q}'")
                    docs = []
        docs = [doc for doc in docs]
    return docs


def manual_entry_abstracts(infile: str, outfile: str) -> pd.DataFrame:
    print("\n---user input of missing abstracts. case-by-case completion.\n")
    try:
        df = pd.read_csv(infile, keep_default_na=False, encoding='utf-8')
    except:
        df = pd.read_csv(infile, keep_default_na=False, encoding='latin1')
    for i in range(len(df)):
        if len(str(df.loc[i, "Abstract Note"])) < 50 or pd.isna(df.loc[i, "Abstract Note"]):
            if isinstance(df.loc[i, "Publication Year"], str):
                print(f'{df.loc[i, "Author"]} {df.loc[i, "Publication Year"].replace(".0","")}')
            else:
                print(f'{df.loc[i, "Author"]} {df.loc[i, "Publication Year"]}')
            print(df.loc[i, "Title"])
            print()
            test = 1
            time.sleep(0.2)
            new_abstract = input("Paste in missing abstract. 's' skips remaining. 'd' deletes previous entry.")
            if new_abstract.lower() == "s":
                df.to_csv(outfile, index=False)
                return df
            if new_abstract.lower() == "d":
                df.loc[i-1, "Abstract Note"] = ""
                print("deleted last. suggesting to rerun manual_entry_abstracts() function.")
                df.to_csv(outfile, index=False)
                return df
            df.loc[i, "Abstract Note"] = new_abstract
            df.to_csv(outfile, index=False)
    return df


def perform_chat_completion(prompt: str, system_message: str, openai_model: str, temperature=0) -> str:
    """
    openai wrapper (internal usage)
    """
    retries = 0
    max_retries = 5
    backoff_time = 5  # Initial backoff time in seconds
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=openai_model,
                messages=[{"role": "system", "content": system_message},
                          {"role": "user", "content": prompt}],
                temperature=temperature)
            if retries > 0:
                print("...success")
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            retries += 1
            print(f"potential openai api issue (overload etc). trying again...({retries}/{max_retries})")
            time.sleep(backoff_time)
            backoff_time *= 2  # Exponential backoff
    return f"API Error {prompt}"



def references_string_as_list(input_string: str) -> list:
    """
    converts string to list if possible via eval()
    used to convert gpt response to python list
    """
    if not ("[" in input_string and "]" in input_string):
        print("no brackets found in extracted references")
        return []
    first_bracket_index = input_string.find('[')
    last_bracket_index = input_string.rfind(']')
    result_string = input_string[first_bracket_index:1+last_bracket_index]
    try:
        evaluated_list = list(eval(result_string))
        return evaluated_list
    except:
        print(f"Error: Could not evaluate the string as a list: {input_string} trimmed: {result_string}.")
        return []


def remove_duplicate_rows(df:pd.DataFrame, similarity_threshold=90):
    """
    for a pd.DataFrame containing literature records
    this function returns dataframe with very similar rows removed
    similarity determined based on Author, Title, and Publication Year
    similarity can be set via similarity_threshold (default 90)
    """
    marked_duplicates = set()
    for outer_index, outer_row in df.iterrows():
        if outer_index in marked_duplicates:
            continue  # Skip rows that have already been marked as duplicates
        outer_author = outer_row['Author']
        outer_title = outer_row['Title']
        outer_year = outer_row['Publication Year']

        for inner_index, inner_row in df.iterrows():
            if outer_index != inner_index and inner_index not in marked_duplicates:
                inner_author = inner_row['Author']
                inner_title = inner_row['Title']
                inner_year = inner_row['Publication Year']
                author_similarity = fuzz.WRatio(outer_author, inner_author)
                title_similarity = fuzz.WRatio(outer_title, inner_title)
                if is_integer(outer_year) and is_integer(inner_year):
                    year_similarity = np.abs(int(inner_year) - int(outer_year))
                else:
                    year_similarity = 0
                if author_similarity > similarity_threshold and title_similarity > similarity_threshold and year_similarity <= 1:
                    marked_duplicates.add(inner_index)
    deduplicated_df = df.drop(index=marked_duplicates)
    deduplicated_df = deduplicated_df.reset_index(drop=True)
    return deduplicated_df


def rename_dict_keys(input_dict: dict, key_mapping: dict) -> dict:
    """
    function to change keys of dict
    """
    new_dict = {}
    for old_key, new_key in key_mapping.items():
        if old_key in input_dict:
            new_dict[new_key] = input_dict[old_key]
    return new_dict


def review(pdffile: str, addonfile = None, lit_csv = None, vectorstore = None, max_paragraphs = 20, paragraph_length=1500, autosearch = True, manual_abstracts = True, upper = 2000, lower = 0, left = 0, openai_model = "gpt-4", max_secondary=50) -> None:
    """
    Use GPT4 to review your science article.

    This function takes a PDF file with a scientific article (and optionally an addonfile i.e. a pdf of a different article with a highly relevant reference section),
    and generates a section-wise review of the PDF document's content. It focuses on potentially relevant/opposing literature.

    Args:
        pdffile (str): The path to the input PDF file to be reviewed. should include a reference list.
        addonfile (str, optional): Additional pdf to guide literature search. pdf file (eg. recent paper or lit review) should include a relevant reference list.
        lit_csv (str, optional): If a zotero style literature corpus already exists as a CSV file, then this path can be used to skip literature search. Also skips addonfile.
        Note that this file needs to have the columns "Author", "Title", "Abstract Note", and "Publication Year"
        vectorstore (str, optional): If you have previously generated a Chroma vectorstore, add the path to the folder. Otherwise, this folder will be generated.
        If the vectorstore argument is provided, literature search and the lit_csv argument will be skipped.
        max_paragraphs (int, optional): The maximum number of paragraphs from the pdffile to review.
        paragraph_length (int, optional): The maximum length (in characters) for each reviewed section in the pdffile.
        upper (int, optional): The number of lines to trim from the top of each page to remove headers. Don't guesstimate but play with trim_pdf() function to set value.
        lower (int, optional): The number of lines to trim from the bottom of each page to remove footers. Don't guesstimate but play with trim_pdf() function to set value.
        left (int, optional): The number of characters to trim from the left of each page (eg to remove line numbers). Don't guesstimate but play with trim_pdf() function to set value.
        autosearch (bool, optional): Whether missing Abstracts should be searched for and imputed.
        manual_abstracts (bool, optional): Whether the user should be prompted to paste in missing abstracts that could not be found.


    Returns:
        None: This function does not return a value. Be aware, it generates files in the working directory.

    Notes:
        - output files will start with the name of pdffile
        - You will need to have a valid OPENAI api key, saved as an environment variable. 
        In your path, the function will look for the value of OPENAI_API_KEY
        - The function assumes that you have a valid chrome driver installed
        - This function takes very long to run (often between thirty and ninety minutes), especially when no literature collection is provided.
        The reason is that various databases will be searched for literature, and the rate limits of various APIs need to be respected.
        
    Example:
        review(pdffile = 'my_paper.pdf', addonfile = "recent_review.pdf")
    """
    ###tests
    if not "OPENAI_API_KEY" in os.environ:
        raise ValueError("could not find OPENAI_API_KEY in your path/environment variables")
        openai.api_key = os.environ['OPENAI_API_KEY']
    test_response = openai.ChatCompletion.create( #API TEST - GPT4
        model=openai_model,
        messages=[{"role": "system", "content": "You say Test"}, {"role": "user", "content": "Say Test"}], 
        temperature=0)
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=chrome_options)
    driver.quit()
    ###end tests

    if not vectorstore:
        if not lit_csv:
            lit_csv = f"""{pdffile.lower().replace(".pdf", "")}_literature.csv"""
            lit_df = generate_literature_csv_from_pdf(pdffile, openai_model, "Author", "Title", "Publication Year", max_secondary=max_secondary, output_path=lit_csv, lower=lower, upper=upper, left=left)
            if addonfile:
                added_lit_df = generate_literature_csv_from_pdf(addonfile, openai_model, "Author", "Title", "Publication Year", max_secondary=max_secondary, output_path=None, include_source_as_ref=True)
                combined_df = pd.concat([added_lit_df, lit_df], ignore_index=True)
                lit_df = remove_duplicate_rows(combined_df).reset_index(drop=True)
        else:
            print("---using pre-existing literature file. reformatting and generating vectorstore")
        if autosearch:
            lit_df = autosearch_missing_abstracts(lit_csv, lit_csv)
        else:
            print("skipping autosearch for missing abstracts")
        if manual_abstracts:
            lit_df = manual_entry_abstracts(lit_csv, lit_csv)
        else:
            print("skipping manual entry of missing abstracts")
        data_formatted = format_texts_from_csv(lit_csv, "Abstract Note", "Author", "Title", "Publication Year")
        vectorstore = generate_vectorstore(data_formatted, lit_csv, max_doc_size = 4000)

    else:
        print("---using pre-existing vectorstore. skipping literature collection.")
    sections = analyze_paragraphs(pdffile, openai_model=openai_model, paragraph_length=paragraph_length, max_paragraphs=max_paragraphs, lower = lower, upper = upper, left = left, verbose=False)
    for i in range(len(sections)):
        if sections[i]["reviewable"]:
            relevant_docs =  get_relevant_docs(sections[i], openai_model=openai_model, chroma_directory=vectorstore)
            if len(relevant_docs) > 0:
                responses, summary = generate_adversarial_texts(sections[i]["page_content"], relevant_docs, openai_model, input_type="text")
                sections[i]["opposition"] = summary
                sections[i]["responses"] = [r["response"] for r in responses]
                sections[i]["references"] = [r["metadata"] for r in responses]
            else:
                sections[i]["opposition"] = "No relevant papers found."
                sections[i]["responses"] = []
                sections[i]["references"] = []
        else:
            sections[i]["opposition"] = "AI deemed section not reviewable."
            sections[i]["responses"] = []
            sections[i]["references"] = []
    create_pdf_from_dicts(sections, f"""{pdffile.lower().replace(".pdf", "")}_review.pdf""")
    return None


def search_with_backoff(query, num_results=1, max_retries=5, retry_delay_base=15):
    """
    search google via googlesearch package. use exponential backoff in case of rate limit
    """
    for retry_count in range(max_retries):
        try:
            results = search(query, num_results)
            if num_results == 1:
                return list(results)[0]
            else:
                return list(results)
        except Exception as e:
            print(f"Attempt {retry_count + 1} failed: {e}")
            retry_delay = retry_delay_base * 2 ** retry_count
            retry_delay += np.random.randint(1, 6)  # Random value between 1 and 5
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    print("Max retries reached, operation failed.")
    return None  # Return None to indicate failure


def shorten(text: str, length: int) -> str:
    """
    string manipulation helper (internal usage)
    """
    if len(text) > length:
        return text[0:length]
    return text


def sort_according_to_abstract_len(temp: pd.DataFrame) -> pd.DataFrame:
    """
    sort dataframe according to length of string in column "Abstract Note"
    """
    temp['Abstract Note'] = temp['Abstract Note'].fillna('')
    temp['Abstract Note Length'] = temp['Abstract Note'].str.len()
    temp = temp.sort_values(by='Abstract Note Length', ascending=True)
    temp = temp.drop(columns=['Abstract Note Length'])
    temp = temp.reset_index(drop=True)
    return temp


def trim_pdf(article_file: str, lower: int, upper: int, left: int, verbose: bool) -> str:
    """
    "experiment with 
    upper (~ 763 ), 
    lower (~ 70 ), 
    and left (~ 10 ) 
    to trim away headers, footers, or line numbers from pdf."

    Returns: string of text from pdf
    
    """
    parts = []
    trimmed_parts = []
    ###START FUNC
    def visitor_body(text, cm, tm, fontDict, fontSize):
        y = tm[5]
        x = tm[4]
        if y > lower and y < upper:
            if x > left:
                parts.append(text)
            else:
                if text != None and text != "\n" and text != "":
                    result = "LEFT" if x < left else  ""
                    text = text.replace("\n", "")
                    trimmed_parts.append(f"CUT {result}: '{text}'")
                    if verbose:
                        print(f"""cut {result} (x = {x}) -> '{text}'""")
        else:
            if text != None and text != "\n" and text != "":
                result = "LOW" if y < lower else "HIGH" if y > upper else ""
                text = text.replace("\n", "")
                trimmed_parts.append(f"CUT {result}: '{text}'")
                if verbose:
                    print(f"""cut {result} (y = {y}) -> '{text}'""")
    ###END FUNC
    pdf_reader = PyPDF2.PdfReader(article_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page.extract_text(visitor_text=visitor_body)
    if len(trimmed_parts) > 0:
        print(f"""With your current settings for upper, lower, and left there were things trimmed from the pdf (try trim_pdf() to finetune pdf cropping). Snippet: [{shorten(",  ".join(trimmed_parts), 800)}]\n\n""")
    else:
        print("TIP: you can use the arguments upper (~763), lower (~70), and left (~10) to trim away headers, footers, or line numbers from pdf. set verbose to True to see what was cut.")
    return "".join(parts)
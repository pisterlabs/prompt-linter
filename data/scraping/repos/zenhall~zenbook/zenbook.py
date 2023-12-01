import os
import re
import PyPDF2
import openai
import glob
import xml.etree.ElementTree as ET
import shutil

gpt_api_key=""




def copy_pdf_to_static(pdf_file_name):
    source_file = f'./book/{pdf_file_name}'
    destination_folder = './static/book/'

    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # copy
    shutil.copy(source_file, destination_folder)








from flask import Flask, render_template
app = Flask(__name__)





from langchain import OpenAI, PromptTemplate 
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
#import openai 




llm = OpenAI(temperature=0,openai_api_base="https://api.chatanywhere.com.cn/v1", openai_api_key=gpt_api_key)


chapter_pages_input=[]
chapter_num=0
# Initialize an empty list to store summaries
summary_list = []
file_name=""


def process_page(page_text):
    match = re.search(r'((\n)Chapter\s+\d+[:\s]+(.+)|第\s+\d+\s+章[:\s]+(.+)|第\s+\d+\s+卷[:\s]+(.+)|第+[一二三四五六七八九十百千万]+章|第+[一二三四五六七八九十百千万]+卷)', page_text, re.IGNORECASE)
    if match:
        start = max(0, match.start() - 10)
        end = min(len(page_text), match.end() + 10)
        context = page_text[start:end]
        return context
    return None

def extract_chapter_pages(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages =  len(reader.pages) 
        chapter_pages = []

        for i in range(num_pages):
            page_text = reader.pages[i].extract_text()
            result = process_page(page_text)
            if result:
                chapter_pages.append((i, result))
        return chapter_pages

def get_pages_from_gpt(chapter_pages):
    


    # 
    openai.api_key = gpt_api_key
    openai.api_base = "https://api.chatanywhere.com.cn/v1"
    
    # make chapter to string
    chapter_pages_info = "\n".join([f"在第 {page_num+1} 页找到如下内容：\n{text}" for page_num, text in chapter_pages])
    # step by step is important
    prompt = "第一步：根据上下文，以下哪些为章节标题,去掉文中引用章节的文本,输出一级章节标题以及页码，使得页码是递增且与出现页码是一一对应的，且每个章节仅存在于一个页码中，第二步，根据上文输出的结果，输出一级标题的页码数组：\n" + chapter_pages_info + "\n"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    print(response['choices'][0]['message']['content'])
    # 解析GPT的回复并获取每一章的页码，这需要你根据GPT实际返回的内容进行调整
    chapter_pages_str = re.search(r'\[\d+(, \d+)*\]', response['choices'][0]['message']['content']).group()
    chapter_pages = [int(page) for page in chapter_pages_str[1:-1].split(', ')]
    
    return chapter_pages

def split_pdf_by_chapters(pdf_path, chapter_pages, output_folder):
    reader = PyPDF2.PdfReader(pdf_path)
    num_pages = len(reader.pages)
    for i, start_page in enumerate(chapter_pages):
            end_page = chapter_pages[i + 1] if i + 1 < len(chapter_pages) else num_pages
            output_path = f'{output_folder}/Chapter_{i + 1}.pdf'

            with open(output_path, 'wb') as output_file:
                writer = PyPDF2.PdfWriter()
                for j in range(start_page-1, end_page):
                    writer.add_page(reader.pages[j])
                writer.write(output_file)


def delete_pdf_files(directory):
    for file in glob.glob(f"{directory}/*.pdf"):
        os.remove(file)


pdf_path = "book/"  # PDF文件的路径
output_folder = "book_buffer"  # 输出文件夹的路径

# 手动分割
def split_pdf_into_chapters(file_path, chapter_pages):
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)

    for i in range(len(chapter_pages)):
        pdf_writer = PyPDF2.PdfWriter()
        start_page = chapter_pages[i] - 1 # 从 0 开始
        end_page = chapter_pages[i + 1] - 1 if i + 1 < len(chapter_pages) else   len(pdf_reader.pages) 

        for page in range(start_page, end_page):
            pdf_writer.add_page(pdf_reader.pages[page])

        output_file_path = f"book_buffer/Chapter_{i+1}.pdf"
        with open(output_file_path, "wb") as out:
            pdf_writer.write(out)

    pdf_file_obj.close()

def summarize_pdf_map(pdf_file_path):

    prompt_template1 = """use chinese to write a concise summary which is less than 70 words of following:


    {text}


    """
    PROMPT1 = PromptTemplate(template=prompt_template1, input_variables=["text"])
    
    prompt_template2 = """use chinese to write a concise summary which is less than 70 words of following:


    {text}


    """
    PROMPT2 = PromptTemplate(template=prompt_template1, input_variables=["text"])


    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=PROMPT1, combine_prompt=PROMPT2)
    summary = chain({"input_documents": docs}, return_only_outputs=True)
    #summary = chain.run(docs)   
    return summary



@app.route('/')
def summaries():
    #global chapter_pages_input,chapter_num
    print(chapter_pages_input)
    print(chapter_num)
    chapter_pages=[chapter_pages_input[chapter_num-1]]
    return render_template('summary_display.html', summary_list=summary_list,chapter_pages=chapter_pages,pdf_file=file_name)
    


    
    
    

if __name__ == "__main__":
    file_name = input("Please enter the name of the pdf file: ")
    mode = input("Please choose a mode (1 ： manual cut   2: auto cut   3 : summarize ): ")
    
    
    # 创建根元素
    root = ET.Element('root')

    
    

    if mode == "1":
        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
            
        delete_pdf_files(output_folder)
        
        chapter_pages = input("Please enter a list of starting page numbers for each chapter, like 9,43,74,: ").split(',')
        # Convert the string of numbers into a list of integers
        split_pdf_into_chapters(pdf_path+file_name, list(map(int,chapter_pages)))
        
        
    elif mode == "2":
        
        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
            
        delete_pdf_files(output_folder)

        # 从PDF中提取章节信息
        chapter_pages = extract_chapter_pages(pdf_path+file_name)
        print(chapter_pages)
        # 使用GPT-3.5-turbo模型获取每一章的页码
        chapter_pages = get_pages_from_gpt(chapter_pages)
        print(chapter_pages)
        # 按章节拆分PDF并保存
        split_pdf_by_chapters(pdf_path+file_name, chapter_pages, output_folder)
        
        # 创建数组元素并将其添加到根元素
        array_element = ET.SubElement(root, file_name)
        
        array_element.text = ' '.join(map(str, chapter_pages))

        # 将根元素保存到 XML 文件
        tree = ET.ElementTree(root)
        tree.write('book.xml')
        
    elif mode == "3":
        
        chapter_num = int(input("Please input chapter number : "))
        
        tree = ET.parse('book.xml')
        # 找到数组元素并将其转换为数组
        root = tree.getroot()
        array_element = root.find(file_name)
        #print(array_element) 
        chapter_pages_input = list(map(int, array_element.text.split()))
        
        

        
        
        summary = summarize_pdf_map(f'{output_folder}/Chapter_{chapter_num}.pdf')

        

        # Add the summary dictionary to the list
        summary_list.append(summary)
            
        # Print the summary list
        print(summary_list)
        
        # 复试文件到static
        copy_pdf_to_static(file_name)
        
        
        while 1:
            app.run(debug=False)







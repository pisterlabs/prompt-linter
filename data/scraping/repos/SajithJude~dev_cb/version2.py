import streamlit as st
from llama_index import GPTVectorStoreIndex, Document, SimpleDirectoryReader, QuestionAnswerPrompt, LLMPredictor, ServiceContext
import json
from langchain import OpenAI
from llama_index import download_loader
from tempfile import NamedTemporaryFile
import base64
import io
import fitz
from PIL import Image
import ast
import os
import glob
PDFReader = download_loader("PDFReader")
import os
import openai 
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from llama_index import download_loader
from xml.etree.ElementTree import Element, SubElement, tostring
import requests
import zipfile
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

from langchain import OpenAI
st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="collapsed")
openai.api_key = os.getenv("OPENAI_API_KEY")


st.title("CourseBot")
st.caption("AI-powered course creation made easy")
DATA_DIR = "data"

PDFReader = download_loader("PDFReader")

loader = PDFReader()


if not os.path.exists("images"):
    os.makedirs("images")

# Create the "pages" folder if it doesn't exist
if not os.path.exists("pages"):
    os.makedirs("pages")



def load_saved_course(course_file):
    with open(course_file, 'r') as infile:
        return json.load(infile)


def call_openai3(source):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=source,
        temperature=0.1,
        max_tokens=3500,
        top_p=1,
        frequency_penalty=0.3,
        presence_penalty=0
    )
    return response.choices[0].text



def call_openai(source):
    messages=[{"role": "user", "content": source}]

    response = openai.ChatCompletion.create(
        model="gpt-4-0314",
        max_tokens=7000,
        temperature=0.1,
        messages = messages
       
    )
    return response.choices[0].message.content

def clear_all_json_files():
    """Clear all JSON files in all directories under the current working directory"""
    
    root_directory = os.path.abspath(os.getcwd())
    
    # Iterate over all files and directories under the root directory
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # Iterate over all files in the current directory
        for filename in filenames:
            # Check if the file has a .json extension
            if filename.endswith('.json'):
                # Open the JSON file, clear its contents, and save the empty file
                file_path = os.path.join(dirpath, filename)
                with open(file_path, 'w') as json_file:
                    json.dump({}, json_file)

def clear_images_folder():
    for file in os.listdir("images"):
        if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            os.remove(os.path.join("images", file))

def clear_pages_folder():
    for file in os.listdir("pages"):
        if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            os.remove(os.path.join("pages", file))


def update_json(topic_data):
    with open("output.json", "w") as f:
        st.session_state.toc = {"Topics": [{k: v} for k, v in topic_data.items()]}
        json.dump({"Topics": [{k: v} for k, v in topic_data.items()]}, f)


def load_db():
    if not os.path.exists("db.json"):
        with open("db.json", "w") as f:
            json.dump({}, f)
    
    with open("db.json", "r") as f:
        db = json.load(f)
    
    return db

def delete_chapter(chapter_name):
    db = load_db()
    if chapter_name in db:
        del db[chapter_name]
        with open("db.json", "w") as f:
            json.dump(db, f)
        return True
    return False

def form_callback(value):
    st.write(value)
    res = st.session_state.index.query("extract all the information belonging to following section into a paragraph "+str(value))
    st.write(res.response)

# def generate_xml_structure(new_dict,coursedesctip,coursedescriptionvoiceover,cn):
#     root = ET.Element("Slides")

#     # First slide with topic names
#     slide = ET.SubElement(root, f"Slide1")
#     slideName = ET.SubElement(slide, "Slide_Name")
#     slideName.text = "Course_Name"
#     crsnmelement =  ET.SubElement(slide, "Course_Name")
#     crsnmelement.text = cn.strip()
#     cd =  ET.SubElement(slide, "Course_Description")
#     cd.text = coursedesctip.strip()
#     cdvo  =  ET.SubElement(slide, "VoiceOver")
#     cdvo1  =  ET.SubElement(cdvo, "VoiceOver_1")
#     cdvo1.text = coursedescriptionvoiceover.strip()


#     slide_counter = 2
#     slide = ET.SubElement(root, f"Slide{slide_counter}")

#     tpcount=1
#     slideName = ET.SubElement(slide, "Slide_Name")
#     slideName.text = "Topics"
#     topic_list = ET.SubElement(slide, "Topics")

#     for topic in new_dict:
#         topic_name = ET.SubElement(topic_list, f"Topic_{tpcount}")
#         topic_name.text = topic
#         tpcount +=1
#     vocount=1
#     voiceovertopic_list = ET.SubElement(slide, "VoiceOver")
#     for topic in new_dict:
#         topic_voiceover = ET.SubElement(voiceovertopic_list, f"VoiceOver_{vocount}")
#         topic_voiceover.text = topic
#         vocount +=1

#     slide_counter += 1

#     # Iterate through topics and subtopics
#     for topic, details in new_dict.items():
#         slide = ET.SubElement(root, f"Slide{slide_counter}")
#         # slideName = ET.SubElement(slide, "Slide_Name")
#         # slideName.text = "Topic_Name"
        
        
#         # Add subtopics if they exist
#         if details["Subtopics"]:
            
#             sub_slide = ET.SubElement(root, f"Slide{slide_counter}")
#             slideName = ET.SubElement(sub_slide, "Slide_Name")
#             slideName.text = "Topic_Name"
#             Topic_Name = ET.SubElement(sub_slide, "Topic_Name")
#             Topic_Name.text= topic
#             subtopiccounter=1
#             for subtopic in details["Subtopics"]:
#                 subtopic_elem = ET.SubElement(sub_slide, f"Subtopic_{subtopiccounter}")
#                 subtopic_elem.text = subtopic["Subtopic"]
#                 subtopiccounter +=1
#             slide_counter += 1

#                 # Add bullets (4 per slide)

#             for subtopic in details["Subtopics"]:
#                 sub_slide = ET.SubElement(root, f"Slide{slide_counter}")
#                 slideName = ET.SubElement(sub_slide, "Slide_Name")
#                 slideName.text = "SubTopic"
#                 Subtopicelement = ET.SubElement(sub_slide, "SubTopic")
            
#             # for subtopic in details["Subtopics"]:
#                 Subtopicelement.text = subtopic["Subtopic"]
#                 bullet_count = 1
#                 bullets_slide = None
#                 for i, bullet in enumerate(subtopic["Bullets"]):
#                     if bullet_count % 4 == 0:
#                         pass 
#                         # bullets_slide = ET.SubElement(sub_slide, "BulletsSlide")
#                     bullet_elem = ET.SubElement(sub_slide, f"Bullet_{bullet_count}")
#                     bullet_elem.text = bullet
#                     bullet_count += 1

#                 vobullet_count = 1
#                 bullets_VO_element = ET.SubElement(sub_slide, "VoiceOver")

#                 for i, bullet in enumerate(subtopic["VoiceOverBullets"]):
#                     if vobullet_count % 4 == 0:
#                         pass
#                     bullet_voiceover_elem = ET.SubElement(bullets_VO_element, f"VoiceOver_{vobullet_count}")
#                     bullet_voiceover_elem.text = bullet
#                     vobullet_count += 1
#                 slide_counter += 1

#         else:
#             slideName = ET.SubElement(slide, "Slide_Name")
#             slideName.text = "Topic_Summary"
#             Topic_Name = ET.SubElement(slide, "Topic_Name")
#             Topic_Name.text= topic
#             Topic_Summary = ET.SubElement(slide, "Topic_Summary")
#             Topic_Summary.text= details["Topic_Summary"].strip()
#             topic_elem = ET.SubElement(slide, "VoiceOver")
#             topic_elem.text = details["VoiceOver"].strip()

#             slide_counter += 1

#     slide = ET.SubElement(root, f"Slide{slide_counter}")
#     slideName = ET.SubElement(slide, "Slide_Name")
#     slideName.text = "Congratulations"
#     messageel =  ET.SubElement(slide, "Message1")
#     messageel.text = "Congratulations"
#     messageel2 =  ET.SubElement(slide, "Message2")
#     messageel2.text = "Congratulations on successful completion of the course."

#     # Generate XML string
#     xml_string = ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8")
#     return xml_string


# import xml.etree.ElementTree as ET

def generate_xml_structure(new_dict,coursedesctip,coursedescriptionvoiceover,cn):
    root = ET.Element("Slides")

    # First slide with topic names
    slide = ET.SubElement(root, f"Slide1")
    slideName = ET.SubElement(slide, "Slide_Name")
    slideName.text = "Course_Name"
    crsnmelement =  ET.SubElement(slide, "Course_Name")
    crsnmelement.text = cn.strip()
    cd =  ET.SubElement(slide, "Course_Description")
    cd.text = coursedesctip.strip()
    cdvo  =  ET.SubElement(slide, "VoiceOver")
    cdvo1  =  ET.SubElement(cdvo, "VoiceOver_1")
    cdvo1.text = coursedescriptionvoiceover.strip()

    slide_counter = 2
    slide = ET.SubElement(root, f"Slide{slide_counter}")

    tpcount=1
    slideName = ET.SubElement(slide, "Slide_Name")
    slideName.text = "Topics"
    # topic_list = ET.SubElement(slide, "Topics")

    for topic in new_dict:
        topic_name = ET.SubElement(slide, f"Topic_{tpcount}")
        topic_name.text = topic
        tpcount +=1
    vocount=1
    voiceovertopic_list = ET.SubElement(slide, "VoiceOver")
    for topic in new_dict:
        topic_voiceover = ET.SubElement(voiceovertopic_list, f"VoiceOver_{vocount}")
        topic_voiceover.text = topic
        vocount +=1

    # Iterate through topics and subtopics
    for topic, details in new_dict.items():
        # Add subtopics if they exist
        slide_counter += 1
        slide = ET.SubElement(root, f"Slide{slide_counter}")
        slideName = ET.SubElement(slide, "Slide_Name")
        slideName.text = "Topic_Name"
        tpname_element = ET.SubElement(slide, "Topic_Name")
        tpname_element.text = topic

        if details["Subtopics"]:
            subtopiccounter=1
            subtopiccounter_1=1
            for subtopic in details["Subtopics"]:
                sp_element = ET.SubElement(slide, f"SubTopic_{subtopiccounter_1}")
                sp_element.text = subtopic["Subtopic"]  
                subtopiccounter_1+=1 
            tpname_vo_element = ET.SubElement(slide, "VoiceOver")
            for subtopic in details["Subtopics"]:
                vo_tag =  ET.SubElement(tpname_vo_element, f"VoiceOver_{subtopiccounter}")
                vo_tag.text = subtopic["Subtopic"]  


            # slide_counter += 1
            for subtopic in details["Subtopics"]:
                slide_counter += 1
                slide = ET.SubElement(root, f"Slide{slide_counter}")
                slideName = ET.SubElement(slide, "Slide_Name")
                slideName.text = "SubTopic"
                Subtopicelement = ET.SubElement(slide, "SubTopic")
                Subtopicelement.text = subtopic["Subtopic"]
                bullet_count = 1
                bullets_slide = None
                for i, bullet in enumerate(subtopic["Bullets"]):
                    if bullet_count % 4 == 0:
                        pass 
                    bullet_elem = ET.SubElement(slide, f"Bullet_{bullet_count}")
                    bullet_elem.text = bullet
                    bullet_count += 1

                vobullet_count = 1
                bullets_VO_element = ET.SubElement(slide, "VoiceOver")

                for i, bullet in enumerate(subtopic["VoiceOverBullets"]):
                    if vobullet_count % 4 == 0:
                        pass
                    bullet_voiceover_elem = ET.SubElement(bullets_VO_element, f"VoiceOver_{vobullet_count}")
                    bullet_voiceover_elem.text = bullet
                    vobullet_count += 1
            #topic summary for subtopic slides

            slide_counter += 1
            slide = ET.SubElement(root, f"Slide{slide_counter}")
            slideName = ET.SubElement(slide, "Slide_Name")
            slideName.text = "Topic_Summary"
            Topic_Name = ET.SubElement(slide, "Topic_Name")
            Topic_Name.text= topic
            Topic_Summary = ET.SubElement(slide, "Topic_Summary")
            Topic_Summary.text= details["Topic_Summary"].strip()
            topic_elem = ET.SubElement(slide, "VoiceOver")
            topic_elem.text = details["VoiceOver"].strip()




        else:
            slide_counter += 1
            slide = ET.SubElement(root, f"Slide{slide_counter}")
            slideName = ET.SubElement(slide, "Slide_Name")
            slideName.text = "Topic_Summary"
            Topic_Name = ET.SubElement(slide, "Topic_Name")
            Topic_Name.text= topic
            Topic_Summary = ET.SubElement(slide, "Topic_Summary")
            Topic_Summary.text= details["Topic_Summary"].strip()
            topic_elem = ET.SubElement(slide, "VoiceOver")
            topic_elem.text = details["VoiceOver"].strip()

    slide_counter += 1
    slide = ET.SubElement(root, f"Slide{slide_counter}")
    slideName = ET.SubElement(slide, "Slide_Name")
    slideName.text = "Congratulations"
    messageel =  ET.SubElement(slide, "Message1")
    messageel.text = "Congratulations"
    messageel2 =  ET.SubElement(slide, "Message2")
    messageel2.text = "Congratulations on successful completion of the course."

    # Generate XML string
    xml_string = ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8")
    # xml_string = xml_string.replace('<?xml version="1.0" ?>', '')
    # st.write(xml_string)
    return xml_string
# Example usage
# xml_output = generate_xml_structure(your_data_structure)
# print(xml_output)


# # Example usage
# xml_output = generate_xml_structure(st.session_state.new_dict)
# print(xml_output)

def process_pdf(uploaded_file):
    loader = PDFReader()
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        documents = loader.load_data(file=Path(temp_file.name))
    
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=3900))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    
    if "index" not in st.session_state:
        index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
        retriever = index.as_retriever(retriever_mode='embedding')
        index = RetrieverQueryEngine(retriever)
        st.session_state.index = index
    # st.session_state.index = index
    return st.session_state.index
        

######################       defining tabs      ##########################################

# upload_col, refine_toc,  extract_col, miss_col, edit_col,voice_col, xml_col, manage_col = st.tabs(["⚪ __Upload Chapter__","⚪ __Refine_TOC__", "⚪ __Extract_Contents__","⚪ __missing_Contents__", "⚪ __Edit Contents__", "⚪ Voice Over__", "⚪ __Export Generated XML__", "⚪ __Manage XMLs__"])
upload_col, refine_toc,  extract_col, voice_col, xml_col = st.tabs(["⚪ __Upload Chapter__","⚪ __Refine_TOC__", "⚪ __Extract_Contents__", "⚪ __Voice Over__", "⚪ __Export Generated XML__"])

if "toc" not in st.session_state:
    st.session_state.toc = {}



######################       Upload chapter column      ##########################################


uploaded_file = upload_col.file_uploader("Upload a Chapter as a PDF file", type="pdf")
# toc_option = upload_col.radio("Choose a method to provide TOC", ("Generate TOC", "Copy Paste TOC"))
forma = """"{
  "Topics": [
    {
      "n.n Topic ": [
        "n.n.n Subtopic ",
        "n.n.n Subtopic ",
      ]
    }
  ]
}

"""
if uploaded_file is not None:
        # clear_all_json_files()

        # index = 
        if "index" not in st.session_state:
            st.session_state.index = process_pdf(uploaded_file)

        upload_col.success("Index created successfully")
        clear_images_folder()
        clear_pages_folder()
    # read PDF file
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # display PDF file
        with fitz.open(uploaded_file.name) as doc:
            for page in doc:  # iterate through the pages
                pix = page.get_pixmap()  # render page to an image
                pix.save("pages/page-%i.png" % page.number) 
            for page_index in range(len(doc)):
                page = doc[page_index]
                image_list = page.get_images(full=True)
                for image_index, img in enumerate(page.get_images(), start=1):
                

                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image = Image.open(io.BytesIO(image_bytes))
                    image_filename = f"images/image_page{page_index}_{image_index}.{image_ext}"
                    image.save(image_filename)


pastecol, copycol = upload_col.columns(2,gap="medium")

copycol.write("AI Generated TOC for unstructured documents")
sampletoc = copycol.button("AI Generated Table")
if sampletoc:
    sample_table = st.session_state.index.query("Generate a table of contents with only sections of topics and subtopics for this book")
    copycol.write("Click on the top right corner to copy, and Paste it on the left, make edits of nessecary and Save")
    copycol.code(sample_table.response)

# elif toc_option == "Copy Paste TOC":
try:
    
    toc_input = pastecol.text_area("Copy the Table of contents from your book and paste them here")

    if pastecol.button("Process and Save"):
        # try:
            # table_of_contents = json.loads(toc_input)
        with st.spinner('Please wait, it might take a while to process the structure'):
            toc_res = "Convert the following table of contents into a json string, use the JSON format given bellow:\n"+ "Table of contents:\n"+ toc_input.strip() + "\n JSON format:\n"+ str(forma) + ". Output should be a valid JSON string."
            str_toc = call_openai(toc_res)
            str_to = str(str_toc)
        # st.write(str_to)
        table_of_contents = json.loads(str_to.strip())

        
        # if "table_of_contents" not in st.session_state:
        st.session_state.table_of_contents = table_of_contents
        pastecol.success("TOC loaded, Go to the next tab")
        pastecol.write(st.session_state.table_of_contents)

except json.JSONDecodeError as e:
    str_toc = call_openai(toc_res)
    table_of_contents = json.loads(str(str_toc))
    st.session_state.table_of_contents = table_of_contents
    pastecol.write(st.session_state.table_of_contents)
    # pastecol.error("Invalid JSON format. Please check your input.")
    pastecol.error(e)




######################       extract content      ##########################################




if "new_dict" not in st.session_state:
    st.session_state.new_dict = {}
for topic in st.session_state.table_of_contents["Topics"]:
    for key, value in topic.items():
        # Add a description for the topic
        st.session_state.new_dict[key] = {'content': '', 'Subtopics': []}
        # Add descriptions for the values
        for item in value:
            st.session_state.new_dict[key]['Subtopics'].append({'content': '', 'Subtopic': item})


pagecol, ecol = extract_col.columns([2,5],gap="large")


for topic_key, topic_value in st.session_state.new_dict.items():
    pagecol.write(f"###### {topic_key}")
    pagecol.button("Extract Topic", key=f"{topic_key}",on_click=form_callback,args=(f"{topic_key}"))
    # expande.write(topic_value["content"])
    for subtopic in topic_value["Subtopics"]:
        expande = pagecol.expander(f"{subtopic['Subtopic']}")
        expande.button("Extract Subtopic", key=f"{subtopic['Subtopic']}",on_click=form_callback, args=(f"{subtopic['Subtopic']}") )
        # expande.write(subtopic["content"])




quer = ecol.button("Extract Contents")
if quer:
    progress_bar = ecol.progress(0)
    total_items = sum(len(subtopics_dict['Subtopics']) for _, subtopics_dict in st.session_state.new_dict.items()) + len(st.session_state.new_dict)
    items_processed = 0
    for topic, subtopics_dict in st.session_state.new_dict.items():
        for subtopic_dict in subtopics_dict['Subtopics']:
            subtopic_name = subtopic_dict['Subtopic']
            subtopicres = st.session_state.index.query("extract all the information under the subtopic  "+str(subtopic_name)+ ", in 4 paragraphs where each paragraph has minimum 40 words.")
            subtopic_dict['content'] = subtopicres.response
            items_processed += 1
            progress_bar.progress(items_processed / total_items)
            ecol.info(f"Extracted {subtopic_name}")
        
        topicres = st.session_state.index.query("extract all the information belonging to following section into a paragraph "+str(topic))
        subtopics_dict['content'] = topicres.response
        items_processed += 1
        progress_bar.progress(items_processed / total_items)


# st.session_state.new_dict = data['data']
for topic_key, topic_value in st.session_state.new_dict.items():
    expander = ecol.expander(f"{topic_key}")
    expander.write(topic_value["content"])
    for subtopic in topic_value["Subtopics"]:
        expander.markdown(f"**{subtopic['Subtopic']}**")
        expander.write(subtopic["content"])


       





######################       voice over      ##########################################




# try:
edcol, excol = voice_col.columns([1,3])

# Course Description
course_description_limit = edcol.number_input("Course Description Word Count Limit", value=30, min_value=1)

# Course Description VoiceOver
course_description_voiceover_limit = edcol.number_input("Course Description VoiceOver Word Count Limit", value=50, min_value=1)

# Topic Summary
topic_summary_limit = edcol.number_input("Topic Summary Word Count Limit", value=30, min_value=1)

# Topic Summary VoiceOver
topic_summary_voiceover_limit = edcol.number_input("Topic Summary VoiceOver Word Count Limit", value=50, min_value=1)

# Number of Bullets per Slide
num_bullets_per_slide = edcol.number_input("Number of Bullets per Slide", value=4, min_value=1)

# Number of Words per Bullet
num_words_bullet = edcol.number_input("Number of Words per Bullet", value=10, min_value=1)

# Bullet VoiceOver
bullet_voiceover_limit = edcol.number_input("VoiceOver per Bullet Word Count Limit", value=20, min_value=1)

# Paraphrasing Percentage Range
# paraphrasing_range = edcol.slider("Paraphrasing % Range", min_value=25, max_value=35, value=(25, 35))

saved_courses = [file for file in os.listdir('.') if file.endswith('.json')]

# Create a select box for saved courses
selectcol,loadcol = excol.columns(2)
cn = excol.text_input("Enter a Course Name")

selected_course = selectcol.selectbox("Select a saved course", saved_courses)
loadcol.write("")
loadcol.write("")

if loadcol.button("Load Project"):
    st.session_state.new_dict = load_saved_course(selected_course)
    excol.success("Project loaded,, you can now continue with Generate XML")
    voice_col.write(st.session_state.new_dict)

gencol, savecol = excol.columns(2)
ex = gencol.button("Generate Voice Over")
# voice_col.write(st.session_state.new_dict)
if ex:
    for topic_key, topic_value in st.session_state.new_dict.items():
        # Add "VoiceOver" key to the main topic
        topic = st.session_state.new_dict[topic_key]
        topic_content = topic['content']
        topic_voiceover_prompt = f"generate a voice over for the following paragraph in {topic_summary_voiceover_limit} words: {topic_content}"
        st.session_state.new_dict[topic_key]["VoiceOver"] = str(call_openai3(topic_voiceover_prompt))
        
        topic_summary_prompt = f"generate a voice over for the following paragraph in {topic_summary_limit} words: {topic_content}"
        st.session_state.new_dict[topic_key]["Topic_Summary"] = str(call_openai3(topic_summary_prompt))
        
        # Check if the topic has subtopics
        # if "Subtopics" in topic_value:
            # Iterate through the subtopics
        for subtopic in topic_value["Subtopics"]:
            subtopic_content = subtopic['content']
            subtopic_content
            subtopic_bullet_prompt = f"Divide the following content :\n {subtopic_content.strip()} \n into {num_bullets_per_slide} unordered bullet points , where each bullet point should have exactly {num_words_bullet} words, The response should be a valid json list of strings."
            bullets = call_openai3(subtopic_bullet_prompt)
            # st.write(bullets)
            bullets
            listbul = ast.literal_eval(bullets.strip())
            subtopic['Bullets'] = listbul
            subtopic_voiceover_prompt = f"By dividing the following content :\n {subtopic_content.strip()} \n Generate {num_bullets_per_slide} voiceover bullet scripts ,where each voiceover bullet script should consist of exactly {bullet_voiceover_limit} words, The response should be a valid json list of strings."
            BulletVoiceOver = call_openai3(subtopic_voiceover_prompt)
            listvoice = ast.literal_eval(BulletVoiceOver.strip())
            subtopic['VoiceOverBullets'] = listvoice


sv_voice = savecol.button("Save voiceover")

if sv_voice:
    json_filename = f"{cn}.json"
    with open(json_filename, 'w') as outfile:
        json.dump(st.session_state.new_dict, outfile)
    # excol.write(st.session_state.new_dict)
if excol.button("generate xml"):
    lsttopics=[]
    for topic in st.session_state.new_dict.keys():
        lsttopics.append(topic)

    course_descriptioninput= f"Generate a course description in exactly {course_description_limit} words for a course containing the following topics:\n"+str(lsttopics)
    coursedesctip = call_openai3(course_descriptioninput)
    course_descriptionvoin= f"Generate a voice over in exactly {course_description_voiceover_limit} words for a course description containing the following topics:\n"+str(lsttopics) +"\n Exclude objectives in the voice over"
    coursedesctipvo = call_openai3(course_descriptionvoin)
    # coursedesctipvo
    # coursedesctip
    # st.session_state.new_dict
    edcol.write(st.session_state.new_dict)
    xml_output = generate_xml_structure(st.session_state.new_dict,coursedesctip,coursedesctipvo,cn)
    pretty_xml = minidom.parseString(xml_output).toprettyxml()
    
    file_name = f"{cn}.xml"
    b64_xml = base64.b64encode(xml_output.encode("utf-8")).decode("utf-8")
    download_button = f'<a href="data:application/xml;base64,{b64_xml}" download="{file_name}">Download XML file</a>'

    # Add the download button
    excol.markdown(download_button, unsafe_allow_html=True)

    excol.code(pretty_xml)





######################       export generated xml      ##########################################


# try:
#     # with 
#     ondu, naduvan, rendu   = xml_col.columns([4,3,4],gap="large")

#     ondu.write("### Select Images")
#     ondu.write("")
#     ondu.write("")

#     left, right = ondu.columns(2)
#     image_topic = left.selectbox("Select a topic", list(st.session_state.new_dict.keys()),label_visibility="collapsed")
#     add_to_topic = right.button("Add Image to Topic")

# # Dropdown menu for selecting a subtopic based on the selected topic
#     image_subtopic = left.selectbox("Select a subtopic", [subtopic["Subtopic"] for subtopic in st.session_state.new_dict[image_topic]["Subtopics"]],label_visibility="collapsed")
#     add_to_subtopic = right.button("Add image to Subtopic")

#     image_files = [f for f in os.listdir("images") if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
#     selected_images = []
#     # for image in image_files:
#     expander = ondu.expander("Select images")
#     n_pages = 20

#     image_exts = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
#     page_index = ondu.number_input("Enter page number", min_value=1, max_value=n_pages, value=1)

#     with ondu.expander(f"Page {page_index}", expanded=True):
#         image_files = [f for f in os.listdir("images") if f.startswith(f'image_page{page_index}_') and f.endswith(tuple(image_exts))]
#         # if image_files:
#         for image_filename in image_files:
#             file_path = os.path.join("images", image_filename)
#             if os.path.isfile(file_path):
#                 ondu.image(file_path, caption=os.path.basename(file_path),width=150)
#             else:
#                 st.warning(f"Image not found: {os.path.basename(file_path)}")
#         # else:
#         #     st.warning("No images found for this page.")
    
#     selected_image = image_filename

#     if add_to_topic:
#         if "img" not in st.session_state.new_dict[image_topic]:
#             st.session_state.new_dict[image_topic]["img"] = []
#         st.session_state.new_dict[image_topic]["img"].append(selected_image)
#         ondu.success(f"Image {selected_image} added to topic {image_topic}")

#     if add_to_subtopic:
#         for subtopic in st.session_state.new_dict[image_topic]["Subtopics"]:
#             if subtopic["Subtopic"] == image_subtopic:
#                 if "img" not in subtopic:
#                     subtopic["img"] = []
#                 subtopic["img"].append(selected_image)
#                 ondu.success(f"Image {selected_image} added to subtopic {image_subtopic}")
#                 break

#     naduvan.write("### Compare ")
#     pages_files = [f for f in os.listdir("pages") if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

#     # if pages_files:
#     selected_page = naduvan.number_input("Compare Images",step=1)
#     selected_image = f"page-{selected_page}.png"
#     # Display the selected image
#     if selected_image:
#         naduvan.image(os.path.join("pages", selected_image), use_column_width=True)
#     else:
#         naduvan.warning("No images found in the 'pages' folder.")




#     rendu.write("### Configure ")
#     # chapter_name = rendu.text_input("enter chapter name")
#     # r1,r2 = rendu.columns(2)

#     # NoOfBullets = r1.text_input("No. of Bullets per Sub Topic")
#     # NoOfWordsPerBullet = r1.text_input("No. of words per Bullet")
#     # NoOfWordsForVOPerBullet = r1.text_input("No. of words for Voice Over per Bullet")
#     save_xml = rendu.button("Save XML")
    


#     if save_xml:

#         # if "edited" not in st.session_state:
#         #     st.session_state.edited = st.session_state.missing
#         #xml_col.write(st.session_state.new_dict)

#         xml_output = json_to_xml(st.session_state.new_dict, chapter_name, NoOfWordsForVOPerBullet, NoOfWordsPerBullet, NoOfBullets) 
#         pretty_xml = minidom.parseString(xml_output).toprettyxml()

#         xml_file_path = os.path.join("images", f"{chapter_name}.xml")
#         with open(xml_file_path, "w") as xml_file:
#             xml_file.write(pretty_xml)
#         # rendu.success(f"XML file saved as {xml_file_path}")

#         with xml_col.expander("XML content"):
#             xml_col.code(pretty_xml)

#         # Zip the entire "images" folder with its contents
#         def zipdir(path, ziph):
#             for root, dirs, files in os.walk(path):
#                 for file in files:
#                     ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), path))

#         zip_file_path = f"images/{chapter_name}.zip"
#         with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#             zipdir("images", zipf)
#         rendu.success(f"Zipped folder saved as {zip_file_path}")

#         # st.session_state.table_of_contents = {}
#         # st.session_state.selected_items = []
#         # st.session_state.new_dict = {}
#         # st.session_state.index = ""
#         # st.session_state.new_dict = {}
 
                
# except (KeyError,NameError, AttributeError) as e:
#     print("Error saving XML")
#     print(f"Error: {type(e).__name__} - {e}")




# # ######################      Manage XML      ##########################################

# # db = load_db()
# # chapter_list = list(db.keys())

# # if chapter_list:

# #     filesinsidefolder = manage_col.selectbox("Select a zip file", [f for f in os.listdir("images") if f.endswith(('.zip'))])

# #     if filesinsidefolder and filesinsidefolder.endswith('.zip'):
# #         file_path = os.path.join("images", filesinsidefolder)
# #         with open(file_path, "rb") as f:
# #             file_bytes = f.read()
# #         manage_col.download_button(
# #             label="Download Zip File",
# #             data=file_bytes,
# #             file_name=filesinsidefolder,
# #             mime="application/zip",
# #         )
   
# #     else:
# #         manage_col.warning("No file selected.")



# #     selected_chapter = manage_col.selectbox("Select a chapter first:", chapter_list)
# #     delete_button = manage_col.button("Delete Chapter")
# #     post_button= manage_col.button("Continue with CourseBOT 2")


# #     if post_button:
# #         url = "https://coursebot2.flipick.com/couresbuilderapi/api/Course/ImportCourse"
# #         payload = json.dumps({
# #                                 "ImportXML": str(db[selected_chapter])
# #                                 })
# #         headers = {
# #                     'Content-Type': 'application/json'
# #                     }


# #         response = requests.request("POST", url, headers=headers, data=payload)
        
# #         print(response)
# #         response_dict = json.loads(response.text)

# #         url_to_launch = response_dict["result"]["urlToLaunch"]
# #         manage_col.subheader("Click on the url bellow to continue.")
# #         manage_col.write(url_to_launch)




# #     if delete_button:
# #         if delete_chapter(selected_chapter):
# #             manage_col.success(f"Chapter {selected_chapter} deleted successfully.")
# #             db = load_db()
# #             chapter_list = list(db.keys())
# #             if chapter_list:
# #                 selected_chapter = manage_col.selectbox("Select a chapter:", chapter_list)
# #                 manage_col.code(db[selected_chapter], language="xml")
# #             else:
# #                 manage_col.warning("No chapters found. Upload a chapter and save its XML first.")
# #         else:
# #             manage_col.error(f"Failed to delete chapter {selected_chapter}.")

# # else:
# #     manage_col.warning("No chapters found. Upload a chapter and save its XML first.")
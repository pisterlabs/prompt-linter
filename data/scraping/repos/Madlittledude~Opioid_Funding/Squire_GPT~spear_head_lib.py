import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os
import re
import streamlit as st
from llama_index.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index import VectorStoreIndex
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
import openai

openai.api_key = os.environ['OPENAI_API_KEY']

@st.cache_data
def ocr_pdf_pages(input_pdf_path, output_text_file):
    # Open the PDF file
    pdf_document = fitz.open(input_pdf_path)

    with open(output_text_file, "w", encoding="utf-8") as text_file:
        for page_number in range(pdf_document.page_count):
            # Extract the page as an image
            page = pdf_document.load_page(page_number)
            image = page.get_pixmap()
            img = Image.frombytes("RGB", (image.width, image.height), image.samples)

            # Perform OCR using pytesseract
            ocr_result = pytesseract.image_to_string(img)

            # Write OCR result to the text file along with the page number
            text_file.write(ocr_result + "\n")
            text_file.write(f"_Page: {page_number + 1}"+ "\n\n")

    # Close the PDF file
    pdf_document.close()

# @st.cache_data
# def ocr_folder(input_folder, output_folder):
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Iterate through all files in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".pdf"):
#             # Generate the output file path
#             input_file_path = os.path.join(input_folder, filename)
#             output_file_path = os.path.join(output_folder, filename.replace(".pdf", ".txt"))

#             # Perform OCR on the PDF and save the text to the output file
#             ocr_pdf_pages(input_file_path, output_file_path)

@st.cache_data
def process_library_text():
    library_dict = {}
    

    for filename in os.listdir('Squire_GPT/Library_TEXT'):
        if filename.endswith('.txt'):
            with open(os.path.join('Squire_GPT/Library_TEXT', filename), 'r', encoding='utf-8') as file:
                content = file.read()

                # Find all occurrences of "Page: {page number}" and split the content accordingly
                pages_split = re.split(r'(_Page:\s*\d+)', content)

                # Create a dictionary for pages, using the page number as the key
                pages_info = {'page_1': pages_split[0].strip()}  # Adding the first page (before the first "Page:" marker)
                for i in range(1, len(pages_split), 2):
                    page_number = int(re.search(r'\d+', pages_split[i]).group()) + 1  # Extracting page number
                    pages_info[f'page_{page_number}'] = pages_split[i + 1].strip()

                library_dict[filename] = {
                    'number_of_pages': len(pages_info),
                    'pages': pages_info}
    return library_dict


def process_chunks(library_dict):
    chunk_dict = {}

    for filename, file_info in library_dict.items():
        pages = file_info['pages']
        file_chunks = {}

        chunk_number = 1

        for i in range(1, file_info['number_of_pages'] + 1):
            page_key = f'page_{i}'
            page_text = pages[page_key]

            # Split the current page into two halves
            split_index = len(page_text) // 2
            top_half = page_text[:split_index]
            bottom_half = page_text[split_index:]

            # Create a full page chunk
            file_chunks[f'chunk{chunk_number}'] = {'text': top_half + bottom_half, 'pages': [i]}
            chunk_number += 0.5

            # If there's a next page, create a half-and-half chunk
            if i < file_info['number_of_pages']:
                next_page_key = f'page_{i + 1}'
                next_page_text = pages[next_page_key]
                next_top_half = next_page_text[:len(next_page_text) // 2]
                file_chunks[f'chunk{chunk_number}'] = {'text': bottom_half + next_top_half, 'pages': [i, i + 1]}
                chunk_number += 0.5

        chunk_dict[filename] = {'number_of_chunks': len(file_chunks), 'chunks': file_chunks}

    return chunk_dict



def to_nodes(chunk_dict):
    nodes = []
    # Iterate through the chunk_dict
    for filename, file_info in chunk_dict.items():
        chunks = file_info['chunks']
        previous_node = None

        for chunk_key, chunk_info in chunks.items():
            text_chunk = chunk_info['text']
            page_numbers = chunk_info['pages']

            # Create a new TextNode for the chunk
            node = TextNode(text=f"Document: {filename}, Pages: {page_numbers}, Text: {text_chunk}")

            # If there's a previous node, set up the NEXT and PREVIOUS relationships
            if previous_node is not None:
                node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=previous_node.node_id)
                previous_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=node.node_id)
                # If needed, you can also include metadata in the relationship
                # previous_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=node.node_id, metadata={"key": "val"})

            nodes.append(node)
            previous_node = node
    return nodes


def calculate_average_word_count(chunk_dict):
    total_word_count = 0
    total_chunks = 0
    
    for filename, file_info in chunk_dict.items():
        chunks = file_info['chunks']
        for chunk_key, chunk_info in chunks.items():
            text = chunk_info['text']
            total_word_count += len(text.split())
            total_chunks += 1

    return total_word_count / total_chunks if total_chunks > 0 else 0


def build_index(nodes, chunk_dict):
    # Calculate the average word count using the chunk dictionary
    average_word_count = calculate_average_word_count(chunk_dict)

    # Determine the value for similarity_top_k based on average_word_count
    if average_word_count < 600:
        similarity_top_k = 5
    else:
        similarity_top_k = 3

    index = VectorStoreIndex(nodes)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.5)
        ]
    )
    return query_engine



def query_index(user_query, query_engine):
    response = query_engine.query(user_query)
    return response



def init_session():
    if 'library_dict' not in st.session_state:
        st.session_state.library_dict = process_library_text()
    if 'chunk_dict' not in st.session_state:
        st.session_state.chunk_dict = process_chunks(st.session_state.library_dict)
    if 'nodes' not in st.session_state:
        st.session_state.nodes = to_nodes(st.session_state.chunk_dict)
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = build_index(st.session_state.nodes, st.session_state.chunk_dict)

def streamlit_interface(query_engine):
    st.title('Search the Opioid Relief Info :pill:')
    user_query = st.text_input('Enter your query:', '')

    # Create a placeholder for the instructions
    instruction_placeholder = st.empty()

    # Display instructions in the placeholder if search has not been performed
    instruction_placeholder.write("Enter your query and press the Search button. Look at the top right, make sure it's running. Don't press Search again until you see it stop running. \nIn your results, you'll find the Response to your query, it's Sources, their Page numbers, and how similar from 0 - 1 your question matched up with the information in the document. ")

    if st.button('Search'):
        # Clear the instructions from the placeholder
        instruction_placeholder.empty()


        if user_query:
            response = query_index(user_query, query_engine)
            st.write('Result:')
            st.write(str(response))
            for source_node in response.source_nodes:
                # Extracting the document name
                document_name_match = re.search(r'Document: (.*?),', source_node.node.text)
                document_name = document_name_match.group(1) if document_name_match else 'N/A'

                # Extracting the pages
                pages_match = re.search(r'Pages: (\[\d+(?:, \d+)*\])', source_node.node.text)
                pages = pages_match.group(1) if pages_match else 'N/A'

                # Extracting the similarity score
                similarity_score = source_node.score

                # Print the results
                st.write(f"Document Name: {document_name}")
                st.write(f"Pages: {pages}")
                st.write(f"Similarity Score: {similarity_score}")
                st.write('---')

        else:
            st.warning('Please enter a valid query.')


def main():

    init_session()
    streamlit_interface(st.session_state.query_engine)

  


if __name__ == "__main__":
    main()







